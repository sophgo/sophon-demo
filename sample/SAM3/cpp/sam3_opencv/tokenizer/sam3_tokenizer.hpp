//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SAM3_TOKENIZER_HPP
#define SAM3_TOKENIZER_HPP

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Byte-pair-encoding tokenizer for SAM3 text prompts.
//
// A C++ port of python/simple_tokenizer.py (CLIP-style BPE): loads the
// gzip-compressed merges table (bpe_simple_vocab_16e6.txt.gz), builds the
// 49408-entry vocabulary and encodes prompts to fixed-length (32) token id
// sequences with <start_of_text>/<end_of_text> framing, zero padding and
// truncation.
//
// Text cleaning is simplified for on-device prompts: ASCII lower-casing and
// whitespace collapsing (ftfy/html.unescape from the python side are skipped,
// non-ASCII letters are treated as letters). For plain ASCII prompts such as
// "a truck" the token ids are identical to the python implementation.
class Sam3Tokenizer {
 public:
  // bpe_gz_path: path of bpe_simple_vocab_16e6.txt.gz
  // context_length: token sequence length (SAM3 uses 32)
  explicit Sam3Tokenizer(const std::string& bpe_gz_path,
                         int context_length = 32);

  // Encode one prompt to [context_length] ids:
  // [sot] + bpe ids + [eot], truncated with last=eot, zero padded.
  std::vector<int32_t> encode(const std::string& text) const;

  int context_length() const { return context_length_; }

 private:
  // Decompress the gzipped BPE vocab (raw deflate via miniz/tinfl).
  static std::string gunzip(const std::string& gz_path);

  // Build byte->unicode-char table and the base 512 vocab entries.
  void buildByteEncoder();

  // Lower-case + collapse whitespace (simplified whitespace_clean().lower()).
  static std::string clean(const std::string& text);

  // CLIP regex scanner: contractions / letter runs / single digits / other runs.
  std::vector<std::string> scan(const std::string& text) const;

  // BPE merge loop on one regex token; returns merged symbols.
  std::vector<std::string> bpe(const std::string& token) const;

  int context_length_;
  std::string bpe_path_;

  // byte (0..255) -> unicode char (UTF-8 encoded)
  std::vector<std::string> byte_encoder_;
  // token string -> id
  std::unordered_map<std::string, int32_t> encoder_;
  // (first, second) -> merge rank
  std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;
  // regex token -> merged symbols (bpe cache)
  mutable std::unordered_map<std::string, std::vector<std::string>> cache_;
};

#endif  // SAM3_TOKENIZER_HPP
