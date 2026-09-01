// capdm — per-layer hidden-state capture for faithful DFlash2 drafter distillation.
//
// Fork of capd with one difference: instead of the post-graph output stream
// (llama_get_embeddings_ith -> n_embd_out), this exports the INPUT residual of a
// chosen transformer block via the fork's public layer-extraction API
// (llama-ext.h: llama_set/get_embeddings_layer_inp). That is the exact stream the
// DFlash2 serve path consumes when building the draft's multi-layer concat
// (common/speculative.cpp ~:1143), so training data matches inference-time inputs.
//
// For mHC targets (qwen4exp) the layer input is the wide hc residual
// (n_embd * hc = 10240 on flash-next); for plain targets it is n_embd.
//
// Output per shard: cap-NNNN.tokens.npy (int32 [T]) + cap-NNNN.layer.npy
// (fp16 [T, W]).
//
// Env: CAPD_TARGET (total tokens), CAPD_SHARD (tokens per shard), CAPD_LAYER
// (block index whose input to export; default n_layer-8).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: llama_set/get_embeddings_layer_inp

#include <clocale>
#include <cstdio>
#include <cstdlib>
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
    const std::string ep = dir + "/cap-" + std::to_string(idx) + ".layer.npy";
    // fwrite(NULL) is a segfault, not an error: fail loudly if the output dir is
    // missing (the 2026-08-31 val-capture crash was exactly this)
    {
        FILE * probe = fopen(dir.c_str(), "r");
        if (!probe) {
            LOG_ERR("%s: output dir '%s' does not exist — create it first\n", __func__, dir.c_str());
            exit(1);
        }
        fclose(probe);
    }
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
    const std::string out_dir = params.out_file.empty() ? "data/capture_layer" : params.out_file;
    const int window     = params.n_batch;     // -b tokens per decode window
    const size_t max_piece_chars = 6000;
    // CAPD_TARGET/CAPD_SHARD keep the arg surface clean; -np = max sequences per window
    const size_t capd_target = std::getenv("CAPD_TARGET") ? strtoul(std::getenv("CAPD_TARGET"), nullptr, 10) : 2000000;
    const size_t capd_shard  = std::getenv("CAPD_SHARD")  ? strtoul(std::getenv("CAPD_SHARD"),  nullptr, 10) : 200000;
    if (chunks_path.empty()) { LOG_ERR("usage: capdm -m model -p chunks.jsonl [-o out_dir] -b window -np seqs; CAPD_TARGET/CAPD_SHARD/CAPD_LAYER env\n"); return 1; }

    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (params.n_parallel <= 1) params.n_parallel = 16;  // sequences per window
    params.n_ctx   = window * 2;  // unified pool holds all window seqs
    params.n_batch = window;
    params.n_ubatch = window;
    params.kv_unified = true;
    params.warmup = false;

    llama_backend_init();
    common_init_result_ptr linit = common_init_from_params(params);
    llama_model * model = linit->model();
    llama_context * ctx = linit->context();
    if (!model || !ctx) { LOG_ERR("%s: failed to load model\n", __func__); return 1; }

    const int n_layer = llama_model_n_layer(model);
    uint32_t layer = n_layer - 8;
    if (const char * e = std::getenv("CAPD_LAYER")) layer = strtoul(e, nullptr, 10);
    if (layer >= (uint32_t) n_layer) { LOG_ERR("%s: CAPD_LAYER %u >= n_layer %d\n", __func__, layer, n_layer); return 1; }

    // request extraction of the chosen block's INPUT residual; the setter marks
    // the scheduler for re-reserve, so the buffers exist from the next decode on.
    llama_set_embeddings_layer_inp(ctx, layer, true);
    LOG_INF("%s: layer input extraction enabled for layer %u of %d\n", __func__, layer, n_layer);

    // read chunks and split
    std::vector<std::string> pieces;
    {
        std::ifstream in(chunks_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.size() < 10) continue;
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
    const int n_embd = llama_model_n_embd(model);
    LOG_INF("%s: layer stream row width n_embd = %d (hc-mean collapsed)\n", __func__, n_embd);

    std::vector<int32_t>  cur_toks;
    std::vector<uint16_t> cur_embd;
    int shard_idx = 0;
    size_t total_tokens = 0;
    const size_t target = capd_target;
    const size_t shard_tokens = capd_shard;

    std::vector<std::vector<llama_token>> pending;   // pieces awaiting a decode window
    size_t pending_tok = 0;
    const int max_seqs = params.n_parallel;
    auto flush_window = [&]() {
        if (pending.empty()) return;
        llama_batch batch = llama_batch_init((int) pending_tok, 0, max_seqs);
        for (size_t s = 0; s < pending.size(); ++s) {
            const auto & toks = pending[s];
            llama_seq_id sid = (llama_seq_id) s;
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { sid }, true);
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed\n", __func__);
            llama_batch_free(batch); pending.clear(); pending_tok = 0; return;
        }
        const float * li = llama_get_embeddings_layer_inp(ctx, layer);
        if (!li) {
            LOG_ERR("%s: layer %u input stream unavailable (extraction not reserved?)\n", __func__, layer);
            llama_batch_free(batch); return;
        }
        // Rows are [n_embd] each after the qwen4exp hc_mean extraction patch
        // (deepseek4.cpp convention); rows map 1:1 to batch token order.
        for (size_t n = 0; n < (size_t) batch.n_tokens; ++n) {
            cur_toks.push_back(batch.token[n]);
            const float * row = li + n * (size_t) n_embd;
            for (int d = 0; d < n_embd; ++d) {
                cur_embd.push_back((uint16_t) float_to_half_bits(row[d]));
            }
        }
        total_tokens += pending_tok;
        llama_batch_free(batch);
        pending.clear(); pending_tok = 0;
        LOG_INF("%s: progress %zu tokens\n", __func__, total_tokens);
        if (cur_toks.size() >= shard_tokens) {
            save_shard(out_dir, shard_idx++, cur_toks, cur_embd, n_embd);
            cur_toks.clear(); cur_embd.clear();
        }
    };

    for (const auto & piece : pieces) {
        if (total_tokens >= target) break;
        auto toks = common_tokenize(vocab, piece, true, false);
        if (toks.empty()) continue;
        if ((int) pending.size() >= max_seqs || pending_tok + toks.size() > (size_t) window) {
            flush_window();
        }
        pending.push_back(std::move(toks));
        pending_tok += pending.back().size();
    }
    flush_window();
    if (!cur_toks.empty()) save_shard(out_dir, shard_idx++, cur_toks, cur_embd, n_embd);
    LOG_INF("%s: done, %zu tokens in %d shards\n", __func__, total_tokens, shard_idx);
    return 0;
}
