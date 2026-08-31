// capd — direct-to-disk hidden-state capture for drafter distillation.
//
// Bypasses the HTTP embedding pipeline entirely: loads the target once, packs
// tokenized text pieces into ~window-token batches (pooling NONE exports the
// wide n_embd_out residual per token after the qwen4exp t_embd fix), and
// writes fp16 .npy shards straight to disk. No JSON, no serialization tax.
//
// Output per shard: cap-NNNN.tokens.npy (int32 [T]) + cap-NNNN.embd.npy (fp16 [T, D]).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// ---- shard writer: .npy v1.0, little-endian ----
static void save_shard(const std::string & dir, int idx,
                       const std::vector<int32_t> & toks,
                       const std::vector<uint16_t> & embd_f16, int n_embd_out) {
    const std::string tp = dir + "/cap-" + std::to_string(idx) + ".tokens.npy";
    const std::string ep = dir + "/cap-" + std::to_string(idx) + ".embd.npy";
    {
        std::string dict = "{'descr': '<i4', 'fortran_order': False, 'shape': (" +
            std::to_string(toks.size()) + ",)}";
        const size_t unpad = 10 + dict.size() + 1;
        dict.append(((unpad + 63) / 64) * 64 - unpad, ' ');
        FILE * f = fopen(tp.c_str(), "wb");
        const unsigned char magic[8] = {0x93,'N','U','M','P','Y',1,0};
        fwrite(magic,1,8,f);
        unsigned short hl = (unsigned short)(((unpad+63)/64)*64 - 10);
        fwrite(&hl,2,1,f); fwrite(dict.c_str(),1,dict.size(),f); fwrite("\n",1,1,f);
        fwrite(toks.data(), 4, toks.size(), f);
        fclose(f);
    }
    {
        std::string dict = "{'descr': '<f2', 'fortran_order': False, 'shape': (" +
            std::to_string(toks.size()) + ", " + std::to_string(n_embd_out) + ")}";
        const size_t unpad = 10 + dict.size() + 1;
        dict.append(((unpad + 63) / 64) * 64 - unpad, ' ');
        FILE * f = fopen(ep.c_str(), "wb");
        const unsigned char magic[8] = {0x93,'N','U','M','P','Y',1,0};
        fwrite(magic,1,8,f);
        unsigned short hl = (unsigned short)(((unpad+63)/64)*64 - 10);
        fwrite(&hl,2,1,f); fwrite(dict.c_str(),1,dict.size(),f); fwrite("\n",1,1,f);
        fwrite(embd_f16.data(), 2, embd_f16.size(), f);
        fclose(f);
    }
    LOG_INF("%s: shard %d written (%zu tokens)\n", __func__, idx, toks.size());
}

// split long text at sentence boundaries, never exceeding max chars
static std::vector<std::string> split_pieces(const std::string & text, size_t max_chars) {
    if (text.size() <= max_chars) return { text };
    std::vector<std::string> pieces;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = std::min(start + max_chars, text.size());
        if (end < text.size()) {
            const size_t dot = text.rfind(". ", end);
            if (dot != std::string::npos && dot > start + max_chars / 2) {
                end = dot + 1; // keep the period
            }
        }
        pieces.push_back(text.substr(start, end - start));
        start = end;
    }
    return pieces;
}

static float float_to_half_bits(float f) {
    // round-to-nearest-even fp32 -> fp16, no denormal flush subtleties needed here
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xff) { // inf/nan
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);      // overflow -> inf
    if (exp <= 0) {                                          // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sign | (man >> shift));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    const std::string chunks_path = params.prompt.empty() ? "" : params.prompt; // -p reused as chunks jsonl
    const std::string out_dir = params.out_file.empty() ? "data/capture" : params.out_file;
    const int window     = params.n_batch;     // -b
    const int shard_size = params.n_parallel;  // -np reused as shard token size (in thousands *1000? no: plain)
    // we reuse -np as shard size in tokens for simplicity
    const size_t max_piece_chars = 6000;
    if (chunks_path.empty()) { LOG_ERR("usage: capd -m model -p chunks.jsonl [-of out_dir] -b window -np shard_tokens\n"); return 1; }

    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.n_ctx   = window;
    params.n_batch = window;
    params.n_ubatch = window;
    params.n_parallel = 1;
    params.kv_unified = true;
    params.warmup = false;

    llama_backend_init();
    common_init_result_ptr linit = common_init_from_params(params);
    llama_model * model = linit->model();
    llama_context * ctx = linit->context();
    if (!model || !ctx) { LOG_ERR("%s: failed to load model\n", __func__); return 1; }

    const int n_embd_out = llama_model_n_embd_out(model);
    LOG_INF("%s: n_embd_out = %d, window = %d\n", __func__, n_embd_out, window);

    // read chunks and split
    std::vector<std::string> pieces;
    {
        std::ifstream in(chunks_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.size() < 10) continue;
            // crude but robust: extract "text": "..." via common json
            nlohmann::json j;
            try { j = nlohmann::json::parse(line); } catch (...) { continue; }
            if (!j.contains("text")) continue;
            for (auto & p : split_pieces(j["text"].get<std::string>(), max_piece_chars)) {
                pieces.push_back(p);
            }
        }
    }
    LOG_INF("%s: %zu pieces loaded\n", __func__, pieces.size());

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<int32_t>  cur_toks;
    std::vector<uint16_t> cur_embd;
    int shard_idx = 0;
    size_t total_tokens = 0;
    const size_t target = (size_t) params.n_predict;   // --n-predict as total target
    const size_t shard_tokens = (size_t) shard_size;

    std::vector<int32_t> pending;   // tokens awaiting a decode window
    auto flush_window = [&]() {
        if (pending.empty()) return;
        llama_batch batch = llama_batch_init(pending.size(), 0, 1);
        for (size_t i = 0; i < pending.size(); ++i) {
            common_batch_add(batch, pending[i], (llama_pos) i, { 0 }, true);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed\n", __func__);
            llama_batch_free(batch);
            pending.clear();
            return;
        }
        for (size_t i = 0; i < pending.size(); ++i) {
            const float * e = llama_get_embeddings_ith(ctx, (int) i);
            GGML_ASSERT(e != nullptr);
            cur_toks.push_back(pending[i]);
            for (int d = 0; d < n_embd_out; ++d) {
                cur_embd.push_back((uint16_t) float_to_half_bits(e[d]));
            }
        }
        total_tokens += pending.size();
        llama_batch_free(batch);
        pending.clear();
        if (total_tokens % (window * 4) < (size_t) window) {
            LOG_INF("%s: progress %zu tokens\n", __func__, total_tokens);
        }
        if (cur_toks.size() >= shard_tokens) {
            save_shard(out_dir, shard_idx++, cur_toks, cur_embd, n_embd_out);
            cur_toks.clear(); cur_embd.clear();
        }
    };

    for (const auto & piece : pieces) {
        if (total_tokens >= target) break;
        auto toks = common_tokenize(vocab, piece, true, false);
        for (auto t : toks) {
            pending.push_back(t);
            if ((int) pending.size() >= window) flush_window();
        }
    }
    flush_window();
    if (!cur_toks.empty()) save_shard(out_dir, shard_idx++, cur_toks, cur_embd, n_embd_out);
    LOG_INF("%s: done, %zu tokens in %d shards\n", __func__, total_tokens, shard_idx);
    return 0;
}
