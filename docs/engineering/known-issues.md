# Problemas conhecidos

O estado vivo fica em `.agent/backlog.json` (`python tools/agent/supervisor.py status`); aqui fica o resumo com as evidências.

## Abertos
| ID | Prioridade | Sintoma | Evidência / reprodução | Situação |
|---|---|---|---|---|
| VKVAL | P1 | Validation layers nos cenários | 22/10, quarto (`--validation`): só `depthClampEnable` = 15 (receitas antigas gravaram 15; normalizado ao carregar e ao criar o pipeline) e avisos de saída de fragmento sem alvo; com a camada o processo não termina no `LARECOMP_EXIT_AT_PRESENT` (o harness espera) | os outros cenários NÃO TESTADOS |
| G1 | P1 | Manchas brancas ao entrar andando no quarto | `docs/DIAGNOSTICO-SWSH.md`; `tests/scenarios/room_enter.json` não reproduz (nem com barrier desligado nem no modo antigo de pular draws) | BLOQUEADO: precisa do caminho do jogador |
| G3 | P1 | Arquibancada piscando no título | só no **modo 60 fps**: a plateia (passe 10) troca de quadro de animação a cada quadro (animação contada em quadros, dobrada pelo modo 60); a 30 fps não ocorre. `tests/regression/title60.json` | não é erro do renderer; ajuste específico do modo 60 (60MODE-ANIM) |
| DYNRES | P2 | O jogo desenha a cena a 80% no mundo aberto (1536x864 em alvo 1920x1080) | F9 em `build/logs/sword-20261018-145705.log` | comportamento do jogo; paliativo: render 1440p |
| PERF60 | P1 | 60 fps a 1440p: P95 / P99 ainda acima da meta | interiores 60; mundo aberto parado ~33 -> 60 fps (`2016be2`, `8a642f1`; `reports/cycles/003.md`); P95 18,9 ms, P99 20,2 ms | render thread corrigida no ciclo 4 (`reports/cycles/004.md`): sem limite, panorâmica 76,5 -> 97,5 fps; resta P95 / P99 |
| HEAPIMPORT | P3 | Hitch de ~250 ms na primeira importação do heap do guest (3,2 GB) para memória global | `reports/cycles/004.md` | fazer no carregamento |
| STAB | P2 | Sessões longas e salvar/carregar sem teste formal | — | NÃO TESTADO |
| LOADS | P3 | Travadas de 200-250 ms ao carregar uma área / um save (tela em fade, poucos draws) | sem capturas de tela (as dos cenários custam ~0,5 s cada e pareciam travadas do jogo): a thread principal espera e uma thread do jogo faz a montagem da cena; 45% dela numa função de seno/cosseno traduzida (`f_70018e6130`, operações de ponto flutuante com checagens de FPCR). `-O2` no jogo todo (21/10): carregamento 228 -> 196-213 ms, 60 fps e folga sem limite iguais, compilação 20 -> 43 min: não adotado | o que resta é o carregamento do próprio jogo; ganhos maiores pedem mudar o código de ponto flutuante que o lifter gera |
| SPEED-UNCAP | P2 | Sem limite de fps (`LARECOMP_FPS=0`) o jogo acelera | o jogo avança a duração fixa da tabela de intervalos por quadro (1/30 s, ou 1/60 s no modo 60): `tests/scenarios/walkspeed*.json` mediu 3x (opção 30 a ~92 fps); a tela de título também é contada em quadros (2x no modo 60, como o G3); a 300-700 fps num carregamento o jogo parou num `svcBreak` | os modos 30 e 60 estão certos (PASSOU); sem limite fica só como medida de folga. Para mais de 60: passo de tempo pelo tempo real (FPS120) |
| FPS120 | P3 | 120 fps | precisa de um passo de tempo ligado ao tempo real (a tabela de intervalos só tem 1/60 s por quadro como menor passo), ver SPEED-UNCAP | depois de 60 estáveis |

## Fechados (com evidência)
| ID | Causa | Correção | Regressão |
|---|---|---|---|
| G2 | `CopyTextureToTexture` com `MIRROR_Y` ignorado | `766239f` | `tests/regression/stairs.json` |
| "posterização 1080p" | Não era bug: a redução da imagem no visualizador | — | — |
| G4 | Programas de shader em cache por **endereço**: o jogo reaproveita a memória de shaders entre áreas e o shader novo recebia a tradução do antigo. Isolado: chave por conteúdo sem receitas = correto; chave por endereço = preto | `80eef24` (programas gráficos e de compute por hash de conteúdo) | `tests/regression/stairs.json` G4-rug-not-black |
| SKIP | Orçamento de 20 ms por quadro e depois pulava; dispatches de compute pulavam sem esperar | `80eef24` (espera com teto por draw + reivindicação do job + receitas) | `SKIP-total` = 0 em stairs e room |
| BARRIER | `CMD_BARRIER` gravado e ignorado | `80eef24` | regressões gerais; A/B sem custo |
| KWAIT | Esperas temporizadas de 1-8 ms duravam ~16 ms (MinGW + Windows) | `27006f3` | `tests/runtime/timed_wait_test.c` |
| TITLE-EXIT | Mortes com código 1: `taskkill /IM lasrecomp.exe` de outro projeto (letsgo `tools/trace_run.sh`) derrubava todo runtime da máquina | `16ee01b` (nosso script) + `scenario.py` repete em EXTERNAL-KILL | — (o script do letsgo-recomp ainda faz isso) |
| TESTCLOCK | Com `LARECOMP_FIXED_TIME` o relógio somava o uptime do PC (o steady clock era o tick do sistema desde o boot do PC), então a hora do jogo, e a luz da cena, mudava entre rodadas | `13a2553` (steady clock = relógio do host na primeira leitura + tick desde então) | `tests/regression/room.json`: 122,7 em duas rodadas seguidas |

## Fora do escopo / NÃO TESTADO
- Pokémon Shield (sem dump).
- Wild Area, batalhas, Dynamax, clima, ciclo dia/noite, menus e inventário, cinemáticas.
