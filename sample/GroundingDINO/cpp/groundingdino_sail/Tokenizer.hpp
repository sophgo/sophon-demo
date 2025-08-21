//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "map"
#include "vector"
#include "string"
#include "fstream"
#include "iostream"

class TokenizerBase
{
protected:
	std::map<std::string, int64> tokenizer_token2idx;
	
public:
	virtual bool load_tokenize(std::string vocab_path) = 0;
	virtual void encode_text(std::string text, std::vector<int64> &idx) = 0;
	std::map<int64, std::string> tokenizer_idx2token;
};

static inline std::string trim(const std::string& s) {
    auto begin = s.begin();
    auto end   = s.end();

    while (begin != end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    if (begin == end) return std::string();

    do { --end; } while (std::isspace(static_cast<unsigned char>(*end)));
    ++end;
    return std::string(begin, end);
}

static inline std::vector<std::string> split_by_space(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) {
        out.push_back(w);
    }
    return out;
}
class TokenizerClip : public TokenizerBase
{
protected:
	std::vector<std::string> stringSplit(const std::string &str, char delim)
	{
		std::vector<std::string> elems;
		auto lastPos = str.find_first_not_of(delim, 0);
		auto pos = str.find_first_of(delim, lastPos);
		while (pos != std::string::npos || lastPos != std::string::npos)
		{
			elems.push_back(str.substr(lastPos, pos - lastPos));
			lastPos = str.find_first_not_of(delim, pos);
			pos = str.find_first_of(delim, lastPos);
		}
		return elems;
	}

	void tokenize(const std::string& text, std::vector<int64_t>& idx) {
		constexpr int64_t CLS_ID = 101;
		constexpr int64_t SEP_ID = 102;

		const int64_t DOT_ID = tokenizer_token2idx.at(".");
		const int64_t UNK_ID = tokenizer_token2idx.at("[UNK]");

		idx.clear();
		idx.push_back(CLS_ID);

		std::vector<std::string> sentences = stringSplit(text, '.');

		for (size_t i = 0; i < sentences.size(); ++i) {
			std::string seg = trim(sentences[i]);
			if (!seg.empty()) {
				std::vector<std::string> words = split_by_space(seg);
				for (const auto& w_raw : words) {
					std::string w = trim(w_raw);
					if (w.empty()) continue;

					auto it = tokenizer_token2idx.find(w);
					if (it != tokenizer_token2idx.end()) {
						idx.push_back(it->second);
					} else {
						idx.push_back(UNK_ID);
					}
				}
			}
			idx.push_back(DOT_ID); // 1012
		}

		idx.push_back(SEP_ID);
	}


public:
	bool load_tokenize(std::string vocab_path) override
	{
		std::ifstream infile;
		infile.open(vocab_path.data());
		if (!infile.good())
		{
			return false;
		}

		std::string s;
		int idx = 0;
		while (getline(infile, s))
		{
			tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
			tokenizer_idx2token.insert(std::pair<int, std::string>(idx, s));
			idx++;
		}
		infile.close();
		return true;
	}

	void encode_text(std::string text, std::vector<int64> &idx) override
	{
		idx.clear();
		return tokenize(text, idx);
	}
};

class TokenizerClipChinese : public TokenizerClip
{
public:
	bool load_tokenize(std::string vocab_path) override
	{
		std::ifstream infile;
		infile.open(vocab_path.data());
		if (!infile.good())
		{
			return false;
		}

		std::string s;
		int idx = 0;
		while (getline(infile, s))
		{
			// printf("%s\n", s.c_str());
			tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
			idx++;
		}
		infile.close();
		return true;
	}

	void encode_text(std::string text, std::vector<int64> &idx) override
	{
#define TOKENIZER_CLS 101
#define TOKENIZER_SEP 102
		idx.clear();
		idx.push_back(TOKENIZER_CLS);
		{
			std::vector<std::string> tokens = stringSplit(text, '.');
			for (auto t : tokens)
			{
				if (tokenizer_token2idx.count(t) > 0)
				{
					idx.push_back(tokenizer_token2idx[t]);
				}
				else
				{
					for (size_t i = 0; i < t.length();)
					{
						int cplen = 1;
						if ((t[i] & 0xf8) == 0xf0)
							cplen = 4; 				// 占用4个字节，前5位为11110
						else if ((t[i] & 0xf0) == 0xe0)
							cplen = 3;				// 占用3个字节，前4位为1110
						else if ((t[i] & 0xe0) == 0xc0)
							cplen = 2;				// 占用2个字节，前3位为110
						// use default
						if ((i + cplen) > t.length())
							cplen = 1;
						auto tmp = t.substr(i, cplen);
						i += cplen;
						idx.push_back(tokenizer_token2idx[tmp]);

						// std::cout << idx[idx.size() - 1] << std::endl;
					}
				}
				idx.push_back(tokenizer_token2idx["."]);
			}
		}
		idx.push_back(TOKENIZER_SEP);
		return;
	}
};

#endif //!TOKENIZER_H