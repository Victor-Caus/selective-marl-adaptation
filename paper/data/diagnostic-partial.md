# Diagnóstico descritivo — sementes 1–3

Valores maiores (menos negativos) são melhores. Sem inferência confirmatória.

| Condição | V2 | LIAM 40M | Δ V2 − LIAM | V2 − sem proteção |
|---|---:|---:|---:|---:|
| none | -8.105 | -15.521 | +7.415 | +0.048 |
| semantic | -18.505 | -19.397 | +0.892 | -0.392 |
| behavioral | -9.007 | -16.083 | +7.076 | +0.066 |
| both | -19.134 | -19.459 | +0.325 | -0.268 |

Não há telemetria de instante de ativação, confiança e rollback nas avaliações LIAM arquivadas. A contagem aplicada mistura correção motora e semântica. Não se pode atribuir causalmente a diferença ao detector a partir destas curvas.

A semântica muda nos dois receptores, mas apenas agent_1 se adapta; o movimento muda apenas em agent_1. A V2 preserva o fluxo emissor, enquanto LIAM aprende ambas as cabeças condicionadas no contexto. Consequentemente, reparar só a recepção não garante reparar a interpretação do parceiro. Isso é uma assimetria verificada no código, mas a magnitude causal requer controles unilaterais/oráculos novos.