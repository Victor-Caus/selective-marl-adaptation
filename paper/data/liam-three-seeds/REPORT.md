# Relatório consolidado — V2 versus LIAM

Estudo encerrado com as três sementes completas em 29/09/2026. A redução de cinco para três foi decidida após inspeção dos resultados, por prazo e interrupções de acesso/execução. O critério confirmatório de cinco sementes NÃO FOI AVALIADO. Não foram descartadas sementes completas por desempenho.

## Recuperação e validação

As três sementes LIAM completaram 40 milhões de passos cada. Foram recuperados os seis checkpoints (3.420.800 e 40.000.000 passos por semente), os logs, configurações, ambientes, protocolo e snapshot do código. Os seis hashes dos checkpoints coincidem com o manifesto do servidor; o snapshot e os atores/memórias locais usados como parceiros também foram conferidos. Os marcadores COMPLETE das três avaliações estão presentes.

Foram validadas 1.680 sessões e 100.800 episódios: chaves completas, retornos finitos, pareamento por cenário e ponto de troca, e médias antes/depois recalculadas dos episódios. Os arquivos recuperados são idênticos aos dados usados nas figuras anteriores. Pesos recuperados são checkpoints para inferência; não incluem todo o estado necessário para retomar treinamento exatamente.

## Resultado principal descritivo

Retorno médio pós-troca; maior (menos negativo) é melhor. Médias por sessão são agregadas dentro de cada semente antes da média e desvio-padrão entre as três sementes.

| Condição | V2 média ± DP | LIAM 40M média ± DP | V2 − LIAM | IC 95% exploratório |
|---|---:|---:|---:|---|
| Sem troca | -8.105 ± 0.139 | -15.521 ± 6.415 | +7.415 | [-8.601; 23.431] |
| Mensagens | -18.505 ± 0.703 | -19.397 ± 2.964 | +0.892 | [-5.512; 7.296] |
| Movimento | -9.007 ± 0.027 | -16.083 ± 5.956 | +7.076 | [-7.778; 21.929] |
| Ambos | -19.134 ± 0.788 | -19.459 ± 2.919 | +0.325 | [-5.897; 6.547] |

Os intervalos usam diferenças pareadas entre apenas três sementes e t de Student com 2 graus de liberdade, sem correção de multiplicidade. Não são testes confirmatórios. Todos os intervalos V2−LIAM40M incluem zero; a média maior não demonstra superioridade estatística.

## Interpretação

- LIAM vence em mensagens e mudanças combinadas nas sementes 1 e 2. Sua semente 3 já apresenta competência nominal fraca, alterando bastante a média. Ela foi preservada.
- V2 fica 7,415 pontos acima de LIAM sem troca e 7,076 pontos acima com mudança motora. Parte importante da diferença vem da competência inicial; a classificação pós-troca não isola adaptação.
- V2 melhora muito sobre o ator congelado no movimento, mas o controle motor-only praticamente iguala V2 nessa condição. Isso sustenta a correção motora, não a necessidade do seletor semântico.
- Sem proteção, a V2 melhora 0,392 ponto em mensagens e 0,268 em mudanças combinadas. A proteção pode impor custo de recuperação, mas faltam registros específicos para atribuir causalidade ao detector.
- V2 não aplicou correção nas 60 sessões sem troca. Isso não implica taxa universal de falso alarme zero. O contador não mede intervenções internas do LIAM.

## Metodologia e limites

Todos os sete métodos controlam somente agent_1 com o mesmo parceiro congelado por semente. São quatro condições, vinte sessões por condição e sessenta episódios por sessão. A mudança ocorre entre os episódios 12 e 35. A semântica afeta ambos os receptores; a rotação afeta somente agent_1. Esta assimetria e o emissor preservado da V2 motivam testes futuros, não uma explicação causal já demonstrada.

LIAM é o núcleo oficial portado ao nosso ambiente, não reprodução numérica do artigo original. O checkpoint longo usa 40M passos; o intermediário usa 3,4208M. Orçamento total, supervisão, memória entre episódios, população de parceiros e número de ambientes paralelos diferem. A campanha anterior de controles simples tinha dois agentes adaptáveis e não deve ser misturada a esta tabela; seu critério de superioridade seletiva falhou.

A equipe de sistemas confirmou ter interrompido processos por uso excessivo durante atividades pedagógicas. Isso confirma a intervenção administrativa nos processos, mas não explica por si só todas as alterações de permissão. Semente 4: execução original interrompida em 24.800.800 passos e tentativa posterior em 7.440.800; ambas excluídas desta comparação. Semente 5 não concluída. Não haverá novos treinos nesta campanha.

## Conclusão e publicação

Há evidência descritiva de correção motora eficaz e preservação do comportamento nominal, mas não demonstração de superioridade geral da adaptação seletiva. O resultado negativo frente a controles simples é parte da contribuição. O PSC e Relay Workshop permanecem proposta de extensão, não ambiente validado nesta pesquisa.

Manuscrito: [LaTeX](../../manuscript-v0.2.tex). Dados, proveniência e comparações acompanham este relatório. Checkpoints e snapshots completos são mantidos separadamente dos arquivos de avaliação.
