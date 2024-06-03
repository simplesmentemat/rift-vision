
# RiftVision

RiftVision é um projeto de visão computacional que utiliza Machine Learning para identificar e extrair dados de replays de partidas de League of Legends. Este projeto é ideal para analistas de jogos, desenvolvedores de ferramentas e entusiastas que desejam obter insights detalhados a partir de vídeos de jogos.

## Funcionalidades

- Detecta objetos no minimapa usando o modelo YOLO.
- Extrai dados de ouro dos jogadores a partir das informações exibidas no vídeo.
- Converte coordenadas de pixel para coordenadas do mapa de jogo.
- Gera um vídeo de saída com as detecções e uma JSON contendo os dados extraídos.

## Requisitos

- Python 3.7+
- OpenCV
- NumPy
- tqdm
- Ultralytics YOLO
- EasyOCR

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/riftvision.git
cd riftvision
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

1. Prepare seu vídeo de entrada e o modelo YOLO treinado.
2. Certifique-se de que as imagens de template (minimapa e tabela de ouro) estejam no diretório `input_video`.
3. Execute o script:

```bash
python riftvision.py --input <caminho_do_video_de_entrada> --output <caminho_do_video_de_saida> --model <caminho_do_modelo_yolo> --frames <numero_de_frames>
```

Parâmetros:
- `--input`: Caminho para o vídeo de entrada.
- `--output`: Caminho para o vídeo de saída.
- `--model`: Caminho para o modelo YOLO.
- `--frames`: Número de frames a serem processados (opcional, padrão é processar o vídeo inteiro).

Exemplo:

```bash
python riftvision.py --input input_video/teste2.mp4 --output output_video/result.mp4 --model model/best.pt --frames 1000
```

## Estrutura do Projeto

```
riftvision/
│
├── app/
│   ├── config/
│   │   ├── minimap.png              # Template da tabela de ouro
│   │   ├── gold_table.png             # Template do minimapa
│   ├── model/
│   │   ├── (modelo YOLO removido devido ao tamanho)
│   ├── processors/
│   │   ├── frame_processor.py
│   │   ├── video_processor.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   ├── template_loader.py
│
├── riftvision.py                  # Script principal do projeto
├── requirements.txt               # Dependências do projeto
└── README.md                      # Instruções e informações sobre o projeto
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorar este projeto.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

Para mais informações, entre em contato com [maatanalista@gmail.co](mailto:maatanalista@gmail.com). Para maior agilidade recomendo usar o discord do [riftvision](https://discord.gg/TqCTsyHF)

---

RiftVision é um projeto de exemplo e não é afiliado ou endossado pela Riot Games. League of Legends é uma marca registrada da Riot Games, Inc. Todos os direitos reservados.
