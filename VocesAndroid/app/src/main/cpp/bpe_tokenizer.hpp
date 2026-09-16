// Reimplementacion en C++ del tokenizador BPE real de XTTS (HF `tokenizers`, formato BPE
// caracter-a-caracter, NO byte-level tipo GPT2 -- confirmado leyendo vocab.json: los tokens base
// de un solo caracter incluyen letras acentuadas del español como entradas propias del vocab,
// ej. 'á','é','ñ','ü' cada una es un token base, no una secuencia de bytes UTF-8 fragmentada).
//
// Carga vocab.tsv/merges.tsv/special_tokens.tsv (volcados desde vocab.json por
// build_corpus_and_dump.py) en tiempo de ejecucion -- NO hardcodea nada del vocabulario, para
// que siga siendo correcto si vocab.json cambia.
//
// Confirmado en Fase 5 (leyendo vocab.json): pre_tokenizer={"type":"Whitespace"},
// normalizer=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False,
// dropout=None. Esto simplifica el algoritmo: BPE estandar sin prefijos/sufijos especiales de
// subpalabra, sin dropout (determinista).
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace xtts_tokenizer {

// --- utilidades UTF-8 ---

inline int Utf8CharLen(unsigned char lead) {
  if ((lead & 0x80) == 0x00) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;  // byte invalido, tratar como 1 (no deberia ocurrir con entrada UTF-8 valida)
}

// Decodifica una string UTF-8 en su secuencia de "caracteres" (cada uno como su propia
// substring UTF-8, ej. 'ñ' -> string de 2 bytes) -- son los "simbolos" base sobre los que opera
// el algoritmo BPE.
inline std::vector<std::string> Utf8Chars(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    int len = Utf8CharLen((unsigned char)s[i]);
    len = std::min(len, (int)(s.size() - i));
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

inline uint32_t Utf8Decode1(const std::string& ch) {
  const unsigned char* b = (const unsigned char*)ch.data();
  if (ch.size() == 1) return b[0];
  if (ch.size() == 2) return ((b[0] & 0x1F) << 6) | (b[1] & 0x3F);
  if (ch.size() == 3) return ((b[0] & 0x0F) << 12) | ((b[1] & 0x3F) << 6) | (b[2] & 0x3F);
  if (ch.size() == 4)
    return ((b[0] & 0x07) << 18) | ((b[1] & 0x3F) << 12) | ((b[2] & 0x3F) << 6) | (b[3] & 0x3F);
  return 0;
}

// Clasificador Unicode acotado a lo necesario para texto en español (y razonablemente para las
// demas lenguas latinas que soporta XTTS) -- ASCII alfanumerico + guion bajo + rango Latin-1
// Supplement/Latin Extended-A de letras acentuadas (u+00C0-u+02AF cubre á-ÿ, ñ, ü, etc.).
// Replica el "\w" Unicode-aware que usa el pretokenizador Whitespace real (crate `regex` de
// Rust) -- no es un match perfecto para TODOS los scripts Unicode (cirilico, arabe, CJK, etc.
// fuera de alcance de este proyecto: idioma objetivo es español).
inline bool IsWordChar(uint32_t cp) {
  if ((cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '_')
    return true;
  if (cp >= 0x00C0 && cp <= 0x02AF) return true;  // Latin-1 Supplement letras + Latin Extended-A/B
  if (cp == 0x00B5) return true;  // micro sign, tratado como letra por algunos motores
  return false;
}

inline bool IsSpaceChar(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v' ||
         cp == 0x00A0;
}

// --- carga de vocab/merges/special tokens ---

// Defensa contra CRLF: `std::ifstream` en modo texto en Windows traduce \r\n->\n de forma
// transparente, pero en Android/Linux NO -- un archivo escrito con saltos de linea de Windows
// deja un '\r' colgando al final de cada linea leida via getline(). Verificado en el dispositivo
// real: sin esto, CADA linea quedaba con un '\r' final corrompiendo toda comparacion de strings
// (0% de coincidencia en vez de 100%). El dump en Python ya se corrigio para escribir '\n'
// explicito, pero esta defensa se deja de todas formas -- cualquier archivo de entrada futuro
// con CRLF no deberia romper esto en silencio.
inline void StripCR(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
}

struct PairHash {
  size_t operator()(const std::pair<std::string, std::string>& p) const {
    return std::hash<std::string>()(p.first) * 1000003u ^ std::hash<std::string>()(p.second);
  }
};

class BpeTokenizer {
 public:
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank;
  std::vector<std::pair<std::string, std::string>> special_tokens;  // ordenados por longitud desc

  bool LoadVocab(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      StripCR(line);
      if (line.empty()) continue;
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      int id = std::stoi(line.substr(0, tab));
      std::string tok = line.substr(tab + 1);
      vocab[tok] = id;
    }
    return true;
  }

  bool LoadMerges(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      StripCR(line);
      if (line.empty()) continue;
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      int rank = std::stoi(line.substr(0, tab));
      std::string pair_str = line.substr(tab + 1);
      auto sp = pair_str.find(' ');
      if (sp == std::string::npos) continue;
      std::string left = pair_str.substr(0, sp);
      std::string right = pair_str.substr(sp + 1);
      // Este vocab multilingue tiene pares DUPLICADOS en la lista de merges (mismo par de
      // caracteres en distintas posiciones). Verificado empiricamente contra el tokenizador
      // real (barrido first-wins vs last-wins sobre 500 frases): last-wins da 99.8% de
      // coincidencia, first-wins solo 19.6% -- se usa la ULTIMA ocurrencia (comportamiento
      // por defecto de construir un mapa sobreescribiendo en el orden del archivo).
      merge_rank[{left, right}] = rank;
    }
    return true;
  }

  bool LoadSpecialTokens(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      StripCR(line);
      if (line.empty()) continue;
      auto tab = line.find('\t');
      if (tab == std::string::npos) continue;
      std::string content = line.substr(tab + 1);
      special_tokens.push_back({content, ""});
    }
    // orden por longitud descendente -- longest-match-first al buscar tokens especiales
    std::sort(special_tokens.begin(), special_tokens.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    return true;
  }

  // Aplica BPE a una sola "palabra" (ya pretokenizada, sin espacios internos), devuelve la lista
  // de subpalabras finales (strings UTF-8).
  std::vector<std::string> BpeWord(const std::string& word) const {
    std::vector<std::string> symbols = Utf8Chars(word);
    // Fusiona UN solo par por iteracion -- el de menor rank (mayor prioridad), el mas a la
    // izquierda en caso de empate -- y vuelve a evaluar desde cero. NO fusiona todas las
    // ocurrencias del par ganador de una vez: verificado empiricamente que esa variante (mas
    // rapida, la implementacion "de libro" de BPE) da resultados distintos a los del
    // tokenizador real de HF en palabras con pares repetidos (ej. "banana": fusionar-todas da
    // "ban"+"ana", fusionar-de-a-uno da "bana"+"na", que es el resultado real).
    while (symbols.size() > 1) {
      int best_rank = -1;
      size_t best_i = 0;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        auto it = merge_rank.find({symbols[i], symbols[i + 1]});
        if (it != merge_rank.end() && (best_rank == -1 || it->second < best_rank)) {
          best_rank = it->second;
          best_i = i;
        }
      }
      if (best_rank == -1) break;
      symbols[best_i] = symbols[best_i] + symbols[best_i + 1];
      symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
  }

  // Pretokenizador "Whitespace" real: separa en runs de \w+ o runs de [^\w\s]+, DESCARTANDO los
  // espacios (no se conservan como tokens). Devuelve la lista de "palabras" a las que luego se
  // les aplica BPE por separado.
  std::vector<std::string> WhitespacePretokenize(const std::string& text) const {
    std::vector<std::string> words;
    std::vector<std::string> chars = Utf8Chars(text);
    std::string current;
    int current_kind = -1;  // 0 = word, 1 = punct/otro no-espacio, -1 = ninguno
    for (auto& ch : chars) {
      uint32_t cp = Utf8Decode1(ch);
      if (IsSpaceChar(cp)) {
        if (!current.empty()) {
          words.push_back(current);
          current.clear();
        }
        current_kind = -1;
        continue;
      }
      int kind = IsWordChar(cp) ? 0 : 1;
      if (kind != current_kind && !current.empty()) {
        words.push_back(current);
        current.clear();
      }
      current += ch;
      current_kind = kind;
    }
    if (!current.empty()) words.push_back(current);
    return words;
  }

  // encode() completo: separa tokens especiales (match literal, longest-first, no solapado),
  // pretokeniza+BPE el resto, mapea todo a ids. Los caracteres base fuera del vocab se mapean a
  // [UNK] (fuse_unk=False: NO se fusionan UNKs consecutivos, confirmado en vocab.json).
  std::vector<int> Encode(const std::string& text) const {
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
      bool matched = false;
      for (auto& st : special_tokens) {
        const std::string& tok = st.first;
        if (text.compare(i, tok.size(), tok) == 0) {
          auto it = vocab.find(tok);
          if (it != vocab.end()) {
            ids.push_back(it->second);
            i += tok.size();
            matched = true;
            break;
          }
        }
      }
      if (matched) continue;
      // acumular hasta el proximo token especial (o fin de texto)
      size_t next_special = text.size();
      for (auto& st : special_tokens) {
        size_t pos = text.find(st.first, i);
        if (pos != std::string::npos && pos < next_special) next_special = pos;
      }
      std::string chunk = text.substr(i, next_special - i);
      i = next_special;
      for (auto& word : WhitespacePretokenize(chunk)) {
        for (auto& piece : BpeWord(word)) {
          auto it = vocab.find(piece);
          if (it != vocab.end()) {
            ids.push_back(it->second);
          } else {
            auto unk = vocab.find("[UNK]");
            if (unk != vocab.end()) ids.push_back(unk->second);
          }
        }
      }
    }
    return ids;
  }
};

}  // namespace xtts_tokenizer
