// Fase 6: segmentacion por oraciones para la app -- Fase 5 confirmo con evidencia (leyendo
// xtts.py, enable_text_splitting=False por defecto y nunca activado por infer_repl.py) que esto
// NO es un requisito de paridad contra ningun codigo de referencia: es una decision de producto
// para la app (sintetizar oracion por oracion, reproducir en streaming para que la latencia
// percibida sea la de la primera oracion, no la del texto completo). Separador simple por
// puntuacion, con el mismo tope duro que ya usa el tokenizador real para español
// (`char_limits["es"] = 239`, tokenizer.py) partiendo por coma o espacio si una oracion sola lo
// excede.
#pragma once
#include <string>
#include <vector>

namespace xtts_tokenizer {

inline std::vector<std::string> SplitSentences(const std::string& text, size_t char_limit = 239) {
  std::vector<std::string> sentences;
  std::string cur;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    cur += c;
    if (c == '.' || c == '!' || c == '?') {
      // no cortar en medio de puntuacion repetida ("...", "?!") ni antes de un digito
      // (posible decimal/abreviatura) -- se deja crecer hasta el ultimo de una racha.
      size_t j = i + 1;
      while (j < text.size() && (text[j] == '.' || text[j] == '!' || text[j] == '?')) {
        cur += text[j];
        ++j;
      }
      if (j < text.size() && (std::isdigit(static_cast<unsigned char>(text[j])))) {
        i = j - 1;
        continue;  // probable decimal/numero, no es fin de oracion real
      }
      i = j - 1;
      sentences.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty()) sentences.push_back(cur);

  // segunda pasada: forzar el tope duro, partiendo por coma o espacio si hace falta
  std::vector<std::string> out;
  for (auto& s : sentences) {
    // trim
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) continue;
    s = s.substr(a, b - a + 1);
    if (s.empty()) continue;

    while (s.size() > char_limit) {
      size_t cut = s.rfind(',', char_limit);
      if (cut == std::string::npos || cut == 0) cut = s.rfind(' ', char_limit);
      if (cut == std::string::npos || cut == 0) cut = char_limit;  // sin espacio/coma, corte duro
      std::string piece = s.substr(0, cut + (s[cut] == ',' ? 1 : 0));
      size_t pa = piece.find_first_not_of(" \t\r\n");
      size_t pb = piece.find_last_not_of(" \t\r\n");
      if (pa != std::string::npos) out.push_back(piece.substr(pa, pb - pa + 1));
      s = s.substr(cut + 1);
      size_t sa = s.find_first_not_of(" \t\r\n");
      s = (sa == std::string::npos) ? "" : s.substr(sa);
    }
    if (!s.empty()) out.push_back(s);
  }
  return out;
}

}  // namespace xtts_tokenizer
