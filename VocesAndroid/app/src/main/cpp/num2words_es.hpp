// Port a C++ de num2words para español -- SOLO los 4 puntos de entrada que realmente usa
// TTS/tts/layers/xtts/tokenizer.py (expand_numbers_multilingual): to_cardinal(int),
// to_cardinal(float), to_ordinal(int), to_currency(float, EUR|USD|GBP). NO es un port de la
// libreria completa.
//
// Fuente portada linea por linea: num2words/base.py (splitnum/clean/to_cardinal_float/
// to_currency genericos), num2words/lang_EU.py (pluralize, set_high_numwords), num2words/
// lang_ES.py (cards, merge, to_ordinal, to_currency override). Todo verificado contra la
// libreria real ejecutandola (no de memoria) antes de portar.
//
// LIMITE EXPLICITO: la libreria real tiene cards hasta "cuatrillón" (10^24), pero eso excede
// int64_t (max ~9.22e18). Esta implementacion solo declara cards hasta "billón" (10^12) --
// suficiente para representar CORRECTAMENTE (via la misma recursion que usa la libreria real)
// cualquier valor hasta justo antes de 10^18, punto en el que la libreria real empezaria a
// preferir la palabra "trillón" en vez de "mil billones". Valores >= 10^18 lanzan
// std::overflow_error en vez de dar un resultado silenciosamente distinto al real.
// char_limits["es"]=239 en el tokenizador real ya acota cuantos digitos son realistas en una
// frase de TTS -- esta cota es enormemente generosa para ese uso.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bpe_tokenizer.hpp"  // Utf8Chars -- para manipular texto por CARACTER, no por byte

namespace xtts_tokenizer {
namespace num2words_es {

// Quita las ultimas `n` CARACTERES Unicode (no bytes) de una string UTF-8 -- replica
// `text[:-n]` de Python, que opera sobre codepoints. Necesario porque "millón" termina en 'ó'
// (2 bytes en UTF-8): un substr por BYTES cortaria a mitad de caracter y daria basura.
inline std::string DropLastNChars(const std::string& s, size_t n) {
  auto chars = Utf8Chars(s);
  if (n >= chars.size()) return "";
  std::string out;
  for (size_t i = 0; i + n < chars.size(); ++i) out += chars[i];
  return out;
}

struct CardEntry { int64_t magnitude; std::string word; };

// Orden de insercion real (num2words/lang_ES.py setup() + lang_EU.py set_high_numwords, con
// GIGA_SUFFIX=None para ES asi que las magnitudes 10^27/21/15/9 NUNCA se insertan -- solo
// MEGA_SUFFIX="illón" en 10^24/18/12/6). splitnum() itera en este orden y toma el primer
// magnitude <= value, o sea el mas grande aplicable -- el orden (de mayor a menor) es lo unico
// que importa.
inline const std::vector<CardEntry>& Cards() {
  static const std::vector<CardEntry> cards = {
      {1000000000000LL, "billón"},
      {1000000LL, "millón"},
      {1000LL, "mil"}, {100LL, "cien"}, {90, "noventa"}, {80, "ochenta"}, {70, "setenta"},
      {60, "sesenta"}, {50, "cincuenta"}, {40, "cuarenta"}, {30, "treinta"},
      {29, "veintinueve"}, {28, "veintiocho"}, {27, "veintisiete"}, {26, "veintiséis"},
      {25, "veinticinco"}, {24, "veinticuatro"}, {23, "veintitrés"}, {22, "veintidós"},
      {21, "veintiuno"}, {20, "veinte"}, {19, "diecinueve"}, {18, "dieciocho"},
      {17, "diecisiete"}, {16, "dieciséis"}, {15, "quince"}, {14, "catorce"}, {13, "trece"},
      {12, "doce"}, {11, "once"}, {10, "diez"}, {9, "nueve"}, {8, "ocho"}, {7, "siete"},
      {6, "seis"}, {5, "cinco"}, {4, "cuatro"}, {3, "tres"}, {2, "dos"}, {1, "uno"}, {0, "cero"},
  };
  return cards;
}

inline std::string CardWord(int64_t magnitude) {
  for (auto& c : Cards())
    if (c.magnitude == magnitude) return c.word;
  throw std::logic_error("magnitud sin card");
}

// --- estructura tipo arbol, igual que las listas/tuplas anidadas de Python splitnum/clean ---
struct Node {
  bool is_leaf = false;
  std::string text;  // leaf
  int64_t num = 0;   // leaf
  std::vector<Node> list;  // no-leaf
  static Node Leaf(std::string t, int64_t n) { Node x; x.is_leaf = true; x.text = std::move(t); x.num = n; return x; }
  static Node List(std::vector<Node> l) { Node x; x.is_leaf = false; x.list = std::move(l); return x; }
};

// merge() de Num2Word_ES (lang_ES.py:274-303), gender_stem="o" fijo (no hay formas femeninas
// para cardinales -- el gender_stem del español para numeros CARDINALES siempre es "o"; la
// alternancia "una libra" es cosa aparte, de to_currency, portada mas abajo).
inline Node MergeEs(const Node& curr, const Node& next) {
  std::string ctext = curr.text;
  int64_t cnum = curr.num;
  std::string ntext = next.text;
  int64_t nnum = next.num;

  if (cnum == 1) {
    if (nnum < 1000000) return next;
    ctext = "un";
  } else if (cnum == 100 && (nnum % 1000 != 0)) {
    ctext += "to";  // "t" + gender_stem("o")
  }

  if (nnum < cnum) {
    if (cnum < 100) return Node::Leaf(ctext + " y " + ntext, cnum + nnum);
    return Node::Leaf(ctext + " " + ntext, cnum + nnum);
  } else if ((nnum % 1000000 == 0) && cnum > 1) {
    ntext = DropLastNChars(ntext, 3) + "lones";
  }

  if (nnum == 100) {
    if (cnum == 5) { ctext = "quinien"; ntext = ""; }
    else if (cnum == 7) { ctext = "sete"; }
    else if (cnum == 9) { ctext = "nove"; }
    ntext += "tos";  // "t" + gender_stem("o") + "s"
  } else {
    ntext = " " + ntext;
  }
  return Node::Leaf(ctext + ntext, cnum * nnum);
}

inline Node SplitNum(int64_t value) {
  if (value >= 1000000000000000000LL)
    throw std::overflow_error("num2words_es: valor fuera del rango soportado (>= 10^18)");
  for (auto& c : Cards()) {
    if (c.magnitude > value) continue;
    int64_t elem = c.magnitude;
    std::vector<Node> out;
    int64_t div, mod;
    if (value == 0) { div = 1; mod = 0; }
    else { div = value / elem; mod = value % elem; }

    if (div == 1) {
      out.push_back(Node::Leaf(CardWord(1), 1));
    } else {
      // rama "tallies" (numeros romanos) de base.py:82-83 nunca se alcanza para cardinales en
      // español con cards[1] presente -- si algun dia se alcanza, fallar en vez de dar un
      // resultado incorrecto silencioso.
      if (div == value) throw std::logic_error("num2words_es: rama 'tallies' inesperada");
      out.push_back(SplitNum(div));
    }
    out.push_back(Node::Leaf(c.word, elem));
    if (mod) out.push_back(SplitNum(mod));
    return Node::List(out);
  }
  throw std::logic_error("num2words_es: SplitNum sin card aplicable");
}

// clean() de base.py:163-182, transliterado 1:1.
inline Node Clean(std::vector<Node> val) {
  while (val.size() != 1) {
    std::vector<Node> out;
    Node& left = val[0];
    Node& right = val[1];
    if (left.is_leaf && right.is_leaf) {
      out.push_back(MergeEs(left, right));
      for (size_t i = 2; i < val.size(); ++i) out.push_back(val[i]);
    } else {
      for (auto& elem : val) {
        if (!elem.is_leaf) {
          if (elem.list.size() == 1) {
            out.push_back(elem.list[0]);
          } else {
            out.push_back(Clean(elem.list));
          }
        } else {
          out.push_back(elem);
        }
      }
    }
    val = std::move(out);
  }
  return val[0];
}

inline std::string ToCardinalInt(int64_t value) {
  std::string out;
  if (value < 0) {
    value = -value;
    out = "menos ";
  }
  // El limite real se aplica dentro de SplitNum (value >= 10^18 lanza overflow_error) -- no
  // hace falta un segundo chequeo aqui (evita ademas un overflow de int64_t al construir un
  // literal "MAXVAL" = 1000*10^18, que ni siquiera cabe en int64_t).
  Node val = SplitNum(value);
  Node cleaned = val.is_leaf ? val : Clean(val.list);
  return out + cleaned.text;
}

// float2tuple + to_cardinal_float de base.py:121-158. Usa std::to_chars (repr de menor longitud
// que redondea exacto al double original) para replicar Decimal(str(value)).as_tuple().exponent
// -- verificado empiricamente (ver Fase 5) que ambos dan la misma representacion "mas corta que
// reproduce el double" para los casos reales de este proyecto (12.5->"12.5", 12.05->"12.05",
// etc.) -- es una propiedad matematica del "shortest round-trip repr", no coincidencia de
// implementacion, asi que se espera que se sostenga en general, no solo en los casos probados.
inline std::string ShortestDecimalRepr(double v) {
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, res.ptr);
}

inline std::string ToCardinalFloat(double value) {
  int64_t pre = (int64_t)value;  // trunca hacia cero, igual que Python int(value) para value>=0
  std::string repr = ShortestDecimalRepr(value);
  size_t dot = repr.find('.');
  std::string frac_digits = (dot == std::string::npos) ? "" : repr.substr(dot + 1);
  int precision = (int)frac_digits.size();

  std::string out = ToCardinalInt(pre);
  if (precision > 0) {
    out += " punto";
    for (char c : frac_digits) {
      out += " " + ToCardinalInt(c - '0');
    }
  }
  return out;
}

// to_cardinal() generico: entero si value es entero, si no cae a float. Nuestro uso real
// (_expand_number/_expand_decimal_point) siempre sabe de antemano si es int u float, asi que se
// exponen ambas funciones por separado en vez de una sola con branching por tipo dinamico.

// to_ordinal() de lang_ES.py:305-349, transliterado 1:1. gender_stem="o" fijo.
inline const std::vector<std::pair<int64_t, std::string>>& OrdWords() {
  static const std::vector<std::pair<int64_t, std::string>> ords = {
      {1, "primer"}, {2, "segund"}, {3, "tercer"}, {4, "cuart"}, {5, "quint"}, {6, "sext"},
      {7, "séptim"}, {8, "octav"}, {9, "noven"}, {10, "décim"}, {20, "vigésim"},
      {30, "trigésim"}, {40, "quadragésim"}, {50, "quincuagésim"}, {60, "sexagésim"},
      {70, "septuagésim"}, {80, "octogésim"}, {90, "nonagésim"}, {100, "centésim"},
      {200, "ducentésim"}, {300, "tricentésim"}, {400, "cuadrigentésim"}, {500, "quingentésim"},
      {600, "sexcentésim"}, {700, "septigentésim"}, {800, "octigentésim"}, {900, "noningentésim"},
      {1000, "milésim"}, {1000000, "millonésim"}, {1000000000, "billonésim"},
      {1000000000000LL, "trillonésim"}, {1000000000000000LL, "cuadrillonésim"},
  };
  return ords;
}

inline std::string OrdWord(int64_t v) {
  for (auto& p : OrdWords())
    if (p.first == v) return p.second;
  throw std::logic_error("num2words_es: sin palabra ordinal para " + std::to_string(v));
}

inline std::string ToOrdinalImpl(int64_t value) {
  if (value == 0) return "";
  std::string text;
  if (value <= 10) {
    text = OrdWord(value) + "o";
  } else if (value <= 12) {
    text = OrdWord(10) + "o" + ToOrdinalImpl(value - 10);
  } else if (value <= 100) {
    int64_t dec = (value / 10) * 10;
    text = OrdWord(dec) + "o " + ToOrdinalImpl(value - dec);
  } else if (value <= 1000) {
    int64_t cen = (value / 100) * 100;
    text = OrdWord(cen) + "o " + ToOrdinalImpl(value - cen);
  } else if (value < 1000000000000000000LL) {
    // dec = 1000 ** int(log(value,1000)) -- potencia de 1000 mas cercana por debajo
    int64_t dec = 1000;
    while (dec * 1000 <= value) dec *= 1000;
    int64_t high_part = value / dec;
    int64_t low_part = value % dec;
    std::string cardinal = (high_part != 1) ? ToCardinalInt(high_part) : "";
    text = cardinal + OrdWord(dec) + "o " + ToOrdinalImpl(low_part);
  } else {
    text = ToCardinalInt(value);
  }
  // strip()
  size_t a = text.find_first_not_of(' ');
  size_t b = text.find_last_not_of(' ');
  if (a == std::string::npos) return "";
  return text.substr(a, b - a + 1);
}

inline std::string ToOrdinalInt(int64_t value) {
  if (value < 0) throw std::domain_error("num2words_es: ordinal negativo no soportado");
  return ToOrdinalImpl(value);
}

// --- to_currency ---
// Formato base (base.py:269-307): "{menos}{cardinal(left)} {plural(left,cr1)} con {cardinal(right)} {plural(right,cr2)}"
// pluralize (lang_EU.py:89-91): forms[0] si n==1, si no forms[1].
// Correcciones especificas de ES (lang_ES.py:355-408) aplicadas DESPUES sobre el string ya
// formado, separando en la parte "antes de con" y "despues de con":
//  - Si currency es femenina (CURRENCIES_UNA): "uno"->"una", "cientos"->"cientas" (solo dollars)
//  - Si cents es femenino (CENTS_UNA): "uno"->"una" (solo cents) [ninguna de EUR/USD/GBP aplica]
//  - Siempre, ambas partes: "veintiuno"->"veintiún", luego "uno"->"un"
struct CurrencyForms { std::string unit_sg, unit_pl, cent_sg, cent_pl; };

inline const CurrencyForms& GetCurrencyForms(const std::string& code) {
  static const CurrencyForms eur{"euro", "euros", "céntimo", "céntimos"};
  static const CurrencyForms usd{"dólar", "dólares", "centavo", "centavos"};
  static const CurrencyForms gbp{"libra", "libras", "penique", "peniques"};
  if (code == "EUR") return eur;
  if (code == "USD") return usd;
  if (code == "GBP") return gbp;
  throw std::invalid_argument("num2words_es: moneda no soportada: " + code);
}

inline bool IsFeminineCurrency(const std::string& code) { return code == "GBP"; }  // libra, entre las soportadas

inline std::string ReplaceAll(std::string s, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

inline std::string Pluralize(int64_t n, const std::string& sg, const std::string& pl) {
  return n == 1 ? sg : pl;
}

// val ya viene como (integer_units, cents) -- el redondeo/parseo (parse_currency_parts, la rama
// float: Decimal quantize .01 ROUND_HALF_UP) se hace ANTES de llamar esto, en el caller, porque
// en nuestro uso real (_expand_currency) el valor de entrada ya es un float de texto parseado
// con como maximo 2 decimales tipicos -- se documenta la simplificacion: redondeo estandar a 2
// decimales (no el ROUND_HALF_UP exacto de Decimal para casos ambiguos de punto flotante, que
// en la practica de textos de TTS no deberia diferir).
inline std::string ToCurrency(double val, const std::string& currency) {
  bool is_negative = val < 0;
  val = std::fabs(val);
  int64_t integer = (int64_t)val;
  int64_t cents = (int64_t)std::llround((val - (double)integer) * 100.0);
  if (cents == 100) { cents = 0; integer += 1; }  // acarreo por redondeo en el borde .995+

  const CurrencyForms& cf = GetCurrencyForms(currency);
  std::string money_str = ToCardinalInt(integer);
  std::string cents_str = ToCardinalInt(cents);
  std::string dollars_part = money_str + " " + Pluralize(integer, cf.unit_sg, cf.unit_pl);
  std::string cents_part = cents_str + " " + Pluralize(cents, cf.cent_sg, cf.cent_pl);

  if (IsFeminineCurrency(currency)) {
    dollars_part = ReplaceAll(dollars_part, "uno", "una");
    dollars_part = ReplaceAll(dollars_part, "cientos", "cientas");
  }
  dollars_part = ReplaceAll(dollars_part, "veintiuno", "veintiún");
  dollars_part = ReplaceAll(dollars_part, "uno", "un");
  cents_part = ReplaceAll(cents_part, "veintiuno", "veintiún");
  cents_part = ReplaceAll(cents_part, "uno", "un");

  std::string result = (is_negative ? "menos " : "") + dollars_part + " con " + cents_part;
  return result;
}

}  // namespace num2words_es
}  // namespace xtts_tokenizer
