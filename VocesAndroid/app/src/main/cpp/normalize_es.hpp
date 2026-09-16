// Fase 5, Stage B: normalizacion de texto en español, portada desde
// TTS/tts/layers/xtts/tokenizer.py (multilingual_cleaners + expand_numbers_multilingual +
// expand_abbreviations_multilingual + expand_symbols_multilingual) y
// TTS/tts/utils/text/cleaners.py (lowercase, collapse_whitespace). Incluye ahora la expansion de
// numeros (num2words_es.hpp) -- ver ese archivo para el alcance/limites del port de num2words.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <regex>
#include <string>
#include <vector>

#include "bpe_tokenizer.hpp"  // Utf8Chars, Utf8Decode1, IsSpaceChar
#include "num2words_es.hpp"

namespace xtts_tokenizer {

// text.replace('"', "") -- TTS/tts/layers/xtts/tokenizer.py:572
inline std::string StripDoubleQuotes(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (c != '"') out.push_back(c);
  }
  return out;
}

// lowercase() = text.lower() -- TTS/tts/utils/text/cleaners.py:54-55. Python str.lower() es
// Unicode-aware; se porta aqui una tabla ACOTADA a ASCII + Latin-1 Supplement/Latin Extended-A
// (cubre español: Á É Í Ó Ú Ñ Ü y las vocales/consonantes acentuadas de otras lenguas latinas
// que XTTS soporta). Fuera de ese rango (cirilico, arabe, CJK, etc.) NO se folding -- documentado,
// no asumido silenciosamente correcto.
inline uint32_t ToLowerCodepoint(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  // Latin-1 Supplement mayusculas U+00C0-U+00DE (salvo U+00D7 multiplicacion, sin minuscula) ->
  // minusculas U+00E0-U+00FE, offset +32, igual que ASCII.
  if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) return cp + 32;
  // Latin Extended-A: mayormente pares consecutivos (par=mayus, impar=minus), U+0100-U+0177.
  if (cp >= 0x0100 && cp <= 0x0177 && (cp % 2 == 0)) return cp + 1;
  return cp;
}

inline void AppendCodepointUtf8(uint32_t cp, std::string& out) {
  if (cp <= 0x7F) {
    out.push_back((char)cp);
  } else if (cp <= 0x7FF) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

inline std::string Lowercase(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (auto& ch : Utf8Chars(text)) {
    AppendCodepointUtf8(ToLowerCodepoint(Utf8Decode1(ch)), out);
  }
  return out;
}

// collapse_whitespace: re.sub(r"\s+", " ", text).strip() -- TTS/tts/utils/text/cleaners.py:58-59.
// Python \s en un patron str es Unicode-aware (incluye NBSP y otros separadores Unicode); se
// cubre aqui el conjunto de espacios que realistamente puede aparecer en texto en español
// (incluido NBSP, U+00A0, frecuente en texto copiado de la web/Word), documentado -- no la tabla
// Unicode White_Space completa.
inline bool IsCollapsibleSpace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v' ||
         cp == 0x00A0 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

inline std::string CollapseWhitespace(const std::string& text) {
  std::string out;
  bool in_space = false;
  for (auto& ch : Utf8Chars(text)) {
    if (IsCollapsibleSpace(Utf8Decode1(ch))) {
      in_space = true;
    } else {
      if (in_space && !out.empty()) out.push_back(' ');
      out += ch;
      in_space = false;
    }
  }
  // strip() final: si el texto terminaba en espacio, in_space queda true pero ya no se agrega
  // nada mas -- el `.strip()` de Python tambien quita espacio inicial, que este bucle ya evita
  // agregando ' ' solo si `!out.empty()`.
  return out;
}

// expand_abbreviations_multilingual(text, "es") -- tokenizer.py:106-118,249-252. Regex real:
// re.compile(f"\\b{abrev}\\.", re.IGNORECASE). Se replica con std::regex ECMAScript + icase.
// AVISO DE PARIDAD (no verificado, documentado): Python `\b` con un patron str es Unicode-aware
// (letras acentuadas cuentan como \w para efectos de limite de palabra); std::regex ECMAScript
// `\b` es ASCII-only. Podria divergir si una abreviatura esta pegada a una vocal acentuada sin
// espacio (caso raro en texto real, no ejercitado por el corpus de 500 frases -- si aparece,
// fallara aqui, no en silencio, porque el propio corpus lo detectaria).
struct AbbrevRule { std::regex pattern; std::string replacement; };

inline std::vector<AbbrevRule> BuildAbbreviationsEs() {
  struct Raw { const char* abbr; const char* full; };
  static const Raw raw[] = {
      {"sra", "señora"}, {"sr", "señor"}, {"dr", "doctor"}, {"dra", "doctora"},
      {"st", "santo"}, {"co", "compañía"}, {"jr", "junior"}, {"ltd", "limitada"},
  };
  std::vector<AbbrevRule> rules;
  for (auto& r : raw) {
    rules.push_back({std::regex(std::string("\\b") + r.abbr + "\\.",
                                 std::regex::ECMAScript | std::regex::icase),
                      r.full});
  }
  return rules;
}

inline std::string ExpandAbbreviationsEs(const std::string& text) {
  static const std::vector<AbbrevRule> rules = BuildAbbreviationsEs();
  std::string out = text;
  for (auto& r : rules) out = std::regex_replace(out, r.pattern, r.replacement);
  return out;
}

// expand_symbols_multilingual(text, "es") -- tokenizer.py:268-279,457-461. Replica la
// particularidad real: dentro del bucle, tras CADA sustitucion de simbolo se hace
// `text.replace("  ", " ")` (una sola pasada de doble-espacio->espacio simple, NO un collapse
// completo -- si quedan 3+ espacios seguidos no se resuelven del todo aqui). Se replica el bug
// tal cual, no se corrige (collapse_whitespace se encarga despues en el pipeline real).
inline std::string ExpandSymbolsEs(const std::string& text) {
  struct Sym { const char* ch; const char* repl; };
  static const Sym syms[] = {
      {"&", " y "}, {"@", " arroba "}, {"%", " por ciento "}, {"#", " numeral "},
      {"$", " dolar "}, {"£", " libra "}, {"°", " grados "},
  };
  std::string out = text;
  for (auto& s : syms) {
    std::string result;
    size_t pos = 0;
    std::string needle = s.ch;
    while (true) {
      size_t found = out.find(needle, pos);
      if (found == std::string::npos) {
        result += out.substr(pos);
        break;
      }
      result += out.substr(pos, found - pos);
      result += s.repl;
      pos = found + needle.size();
    }
    out = result;
    // replace("  ", " ") -- Python str.replace hace TODAS las ocurrencias no solapadas de "  "
    // en UN solo barrido izquierda-a-derecha (no relee el resultado ya sustituido) -- una sola
    // pasada por simbolo, no un collapse completo (si quedan 3+ espacios seguidos no se
    // resuelven del todo aqui; se replica el comportamiento real, no se corrige).
    std::string tmp;
    size_t p = 0;
    while (true) {
      size_t f = out.find("  ", p);
      if (f == std::string::npos) {
        tmp += out.substr(p);
        break;
      }
      tmp += out.substr(p, f - p) + " ";
      p = f + 2;
    }
    out = tmp;
  }
  // .strip() final de expand_symbols_multilingual
  size_t start = out.find_first_not_of(' ');
  size_t end = out.find_last_not_of(' ');
  if (start == std::string::npos) return "";
  return out.substr(start, end - start + 1);
}

// re.sub(pattern, callback, text) generico -- reemplaza cada match (izquierda a derecha, no
// solapados, mismo orden que Python) por callback(match). Necesario porque std::regex_replace
// no soporta un callback, solo un string de formato.
inline std::string RegexSubCallback(const std::string& text, const std::regex& re,
                                     const std::function<std::string(const std::smatch&)>& fn) {
  std::string out;
  auto begin = std::sregex_iterator(text.begin(), text.end(), re);
  auto end = std::sregex_iterator();
  size_t last = 0;
  for (auto it = begin; it != end; ++it) {
    const std::smatch& m = *it;
    out += text.substr(last, m.position(0) - last);
    out += fn(m);
    last = m.position(0) + m.length(0);
  }
  out += text.substr(last);
  return out;
}

// _dot_number_re = r"\b\d{1,3}(.\d{3})*(\,\d+)?\b" -- tokenizer.py:489. El punto SIN escapar es
// un bug real del codigo original (matchea CUALQUIER caracter, no solo '.') -- se replica tal
// cual, no se corrige (ver aviso del advisor: "port the buggy behavior verbatim").
inline const std::regex& DotNumberRe() {
  static const std::regex re(R"(\b\d{1,3}(.\d{3})*(\,\d+)?\b)");
  return re;
}

inline std::string RemoveDots(const std::string& s) {
  std::string out;
  for (char c : s) if (c != '.') out.push_back(c);
  return out;
}

// _currency_re -- tokenizer.py:482-486.
inline const std::regex& CurrencyReGbp() { static const std::regex re(R"(((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£)))"); return re; }
inline const std::regex& CurrencyReUsd() { static const std::regex re(R"(((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$)))"); return re; }
inline const std::regex& CurrencyReEur() { static const std::regex re(R"((([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+))))"); return re; }

// Python float(s) exige que TODA la string sea un float valido, o lanza ValueError.
// std::stod, en cambio, parsea el PREFIJO valido mas largo y sigue de largo silenciosamente --
// una divergencia real encontrada al validar contra el corpus (caso "$1,234.56": tras
// normalizar separadores queda "1.234.56", con DOS puntos; Python float() lanza ValueError ahi
// -> el try/except de expand_numbers_multilingual lo atrapa y dejar el texto sin tocar en ese
// paso; std::stod en cambio parseaba "1.234" y seguia de largo, dando un resultado
// silenciosamente distinto). Se replica el "todo o nada" de Python explicitamente.
inline double StrictStod(const std::string& s) {
  if (s.empty()) throw std::invalid_argument("StrictStod: string vacia");
  size_t consumed = 0;
  double v = std::stod(s, &consumed);
  if (consumed != s.size()) throw std::invalid_argument("StrictStod: sobra texto tras el numero: " + s);
  return v;
}

// _expand_currency -- tokenizer.py:512-539 (rama "es": and_equivalents["es"]=" con ").
inline std::string ExpandCurrencyMatch(const std::string& matched, const std::string& currency_code) {
  std::string s = matched;
  // .replace(",", ".") luego re.sub(r"[^\d.]", "", ...) -- se queda solo con digitos y puntos.
  for (auto& c : s) if (c == ',') c = '.';
  std::string digits_only;
  for (char c : s) if ((c >= '0' && c <= '9') || c == '.') digits_only.push_back(c);
  double amount = digits_only.empty() ? 0.0 : StrictStod(digits_only);

  std::string full = xtts_tokenizer::num2words_es::ToCurrency(amount, currency_code);

  // amount.is_integer(): truncar en la ULTIMA ocurrencia de " con " (tokenizer.py:534-537).
  double int_part;
  if (std::modf(amount, &int_part) == 0.0) {
    size_t pos = full.rfind(" con ");
    if (pos != std::string::npos) full = full.substr(0, pos);
  }
  return full;
}

// _decimal_number_re = r"([0-9]+[.,][0-9]+)" -- tokenizer.py:490.
inline const std::regex& DecimalNumberRe() {
  static const std::regex re(R"(([0-9]+[.,][0-9]+))");
  return re;
}

// _ordinal_re["es"] = r"([0-9]+)(º|ª|er|o|a|os|as)" -- tokenizer.py:466. SIN \b (a proposito --
// el codigo real matchea a mitad de palabra, ej. "5ta" solo matchea el "5" si "ta" no encaja en
// el grupo de sufijos, dejando "ta" suelto -- se replica tal cual).
inline const std::regex& OrdinalReEs() {
  static const std::regex re(u8"([0-9]+)(º|ª|er|o|a|os|as)");
  return re;
}

// _number_re = r"[0-9]+" -- tokenizer.py:481.
inline const std::regex& NumberRe() {
  static const std::regex re(R"([0-9]+)");
  return re;
}

// expand_numbers_multilingual(text, "es") -- tokenizer.py:550-568, rama lang!="zh". Orden real:
// 1) quitar puntos de separador de miles, 2) moneda GBP/USD/EUR, 3) decimales (si lang!="tr"),
// 4) ordinales, 5) cardinales sueltos.
inline std::string ExpandNumbersEs(const std::string& text) {
  std::string t = RegexSubCallback(text, DotNumberRe(), [](const std::smatch& m) {
    return RemoveDots(m.str(0));
  });

  try {
    t = RegexSubCallback(t, CurrencyReGbp(), [](const std::smatch& m) { return ExpandCurrencyMatch(m.str(0), "GBP"); });
    t = RegexSubCallback(t, CurrencyReUsd(), [](const std::smatch& m) { return ExpandCurrencyMatch(m.str(0), "USD"); });
    t = RegexSubCallback(t, CurrencyReEur(), [](const std::smatch& m) { return ExpandCurrencyMatch(m.str(0), "EUR"); });
  } catch (...) {
    // tokenizer.py:558-563: except: pass -- se ignora cualquier fallo de esta etapa completa.
  }

  t = RegexSubCallback(t, DecimalNumberRe(), [](const std::smatch& m) {
    std::string amount = m.str(1);
    for (auto& c : amount) if (c == ',') c = '.';
    double v = std::stod(amount);
    return xtts_tokenizer::num2words_es::ToCardinalFloat(v);
  });

  t = RegexSubCallback(t, OrdinalReEs(), [](const std::smatch& m) {
    int64_t v = std::stoll(m.str(1));
    return xtts_tokenizer::num2words_es::ToOrdinalInt(v);
  });

  t = RegexSubCallback(t, NumberRe(), [](const std::smatch& m) {
    int64_t v = std::stoll(m.str(0));
    return xtts_tokenizer::num2words_es::ToCardinalInt(v);
  });

  return t;
}

// Pipeline compuesto SIN numeros (para debugging/Stage B aislado). Orden real tomado de
// multilingual_cleaners (tokenizer.py:571-582): comillas -> minusculas -> [numeros] ->
// abreviaturas -> simbolos -> espacios.
inline std::string NormalizeEsNoNumbers(const std::string& text) {
  std::string t = StripDoubleQuotes(text);
  t = Lowercase(t);
  t = ExpandAbbreviationsEs(t);
  t = ExpandSymbolsEs(t);
  t = CollapseWhitespace(t);
  return t;
}

// Pipeline completo = multilingual_cleaners(text, "es") -- tokenizer.py:571-582.
inline std::string NormalizeEs(const std::string& text) {
  std::string t = StripDoubleQuotes(text);
  t = Lowercase(t);
  t = ExpandNumbersEs(t);
  t = ExpandAbbreviationsEs(t);
  t = ExpandSymbolsEs(t);
  t = CollapseWhitespace(t);
  return t;
}

}  // namespace xtts_tokenizer
