# MLOps End-to-End Pipeline

Este repositório contém uma solução completa e integrada de **MLOps (Machine Learning Operations)** para automatizar todo o ciclo de vida de modelos de aprendizado de máquina, desde a ingestão e processamento de dados até o deploy e monitoramento em produção.

O objetivo do projeto é demonstrar práticas de engenharia de software aplicadas à Ciência de Dados, garantindo reprodutibilidade, governança e escalabilidade de workloads de IA.

## 🛠️ Arquitetura e Tecnologias

A arquitetura do pipeline é composta pelas seguintes camadas e ferramentas:

*   **Linguagem Core:** Python
*   **Orquestração de Workflows:** Apache Airflow / Prefect (Agendamento e execução de DAGs de dados)
*   **Processamento e Feature Store:** Pandas / PySpark (Manipulação eficiente de dados)
*   **Rastreamento e Governança:** MLflow (Registro de experimentos, parâmetros, métricas e artefatos de modelos)
*   **Containerização:** Docker & Docker Compose (Isolamento completo do ambiente de desenvolvimento e produção)
*   **Qualidade e Validação:** Great Expectations / Pydantic (Validação de schemas e dados de entrada)

## 🚀 Como Executar o Projeto

### Pré-requisitos
*   Docker e Docker Compose instalados na máquina.
*   Python 3.10+ (para desenvolvimento local fora do container).

### Passo a Passo

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com
   cd mlops-end-to-end-pipeline
   ```

2. **Configurar as Variáveis de Ambiente:**
   Copie o arquivo de exemplo e preencha as credenciais necessárias:
   ```bash
   cp .env.example .env
   ```

3. **Subir o Ambiente via Docker:**
   Inicialize todos os serviços do ecossistema (Airflow, MLflow Tracker, Banco de Dados, etc.):
   ```bash
   docker-compose up --build -d
   ```

4. **Acessar as Interfaces Virtuais:**
   *   **MLflow UI:** `http://localhost:5000`
   *   **Airflow Webserver:** `http://localhost:8080`

## 📁 Estrutura do Repositório

```text
├── .github/               # Pipelines de CI/CD (GitHub Actions)
├── config/                # Arquivos de configuração (.yaml, .ini)
├── dags/                  # Pipelines de orquestração do Airflow
├── src/                   # Código-fonte principal
│   ├── data/              # Scripts de ingestão e processamento (ETL)
│   ├── features/          # Engenharia de atributos (Feature Engineering)
│   ├── models/            # Scripts de treino, avaliação e inferência
│   └── utils/             # Funções utilitárias e helpers
├── tests/                 # Testes unitários e de integração
├── docker-compose.yml     # Orquestração dos containers
├── Dockerfile             # Definição da imagem do ambiente de ML
└── requirements.txt       # Dependências de bibliotecas Python
```
