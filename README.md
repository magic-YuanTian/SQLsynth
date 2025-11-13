# SQLsynth

This is the repo for IUI'25 paper, [Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation](https://arxiv.org/abs/2502.15980).

*Note: This repo serves as the latest and backup version of the [official repo](https://github.com/adobe/nl_sql_analyzer).*
  

**SQLsynth** is both an interactive annotation and automated system designed for generating *schema-specific* text-to-SQL datasets. 


### Key features:

- PCFG-based SQL sampler (probabilities and rules are configurable)
- Use grammar to parse SQL into step-by-step NL explanations
- SQL-to-text generation based on in-context leanring & step-by-step explanation
- Novel alignment feature: Aligning SQL to NL by step-by-step explanations
- Dataset statistics & visualization  


## 🌟 Features

- **SQL Query Synthesis**: Automatically generate diverse SQL queries based on database schemas using probabilistic context-free grammar (PCFG)
- **SQL-to-NL Translation**: Convert SQL queries into natural language descriptions using hybrid rule-based and LLM approaches
- **NL-to-SQL Analysis**: Analyze alignment between natural language questions and SQL query components
- **Database Record Generation**: Synthesize realistic database records respecting foreign key constraints and data types
- **Dataset Analysis**: Comprehensive statistical analysis of SQL query datasets
- **Interactive UI**: User-friendly web interface with schema design, database management, and analysis capabilities
- **In-Context Learning**: Leverage similar examples from the Spider dataset for improved natural language generation
- **Batch Processing**: Script-based data synthesis for generating large-scale datasets

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Script-Based Synthesis](#script-based-synthesis)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [User Study](#user-study)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SQLsynth.git
cd SQLsynth
```

2. Install Python dependencies:
```bash
cd backend
pip install flask
pip install flask_cors
pip install sql-metadata
pip install openai
pip install nltk
pip install spacy
pip install sqlparse
python -m spacy download en_core_web_sm
```

3. Configure LLM API:
   - Open `backend/openai_api.py`
   - Implement your own `get_openai_response()` function
   - The function should take a string prompt as input and return a string response


### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. If you encounter missing dependencies, please use `npm install` for necessary packages based on pop-up instructions.

## 🎯 Quick Start

### Running the Application

1. **Start the Backend Server**:
```bash
cd backend
python server.py
```
The backend will run on `http://localhost:5001` by default.

2. **Start the Frontend**:
```bash
cd frontend
npm start
```
The frontend will run on `http://localhost:3000` by default.

3. Open your browser and navigate to `http://localhost:3000`

4. Enjoy it! 🎉

### Basic Workflow

1. **Schema Tab**: Design or import your database schema
2. **Database Tab**: Generate synthetic records for your schema
3. **Dataset Tab**: Synthesize SQL queries and natural language pairs
4. **Analysis Tab**: Analyze alignment between SQL and natural language

## 🏗️ Architecture

### Backend (`backend/`)

- **`server.py`**: Flask server handling all API endpoints
- **`SQL_synthesizer.py`**: PCFG-based SQL query generation
- **`SQL2NL_clean.py`**: Rule-based SQL decomposition and explanation
- **`llm_analysis.py`**: LLM prompts and analysis functions
- **`records_synthesizer.py`**: Database record generation with constraint satisfaction
- **`ICL_retriever.py`**: In-context learning example retrieval
- **`db_handling.py`**: Database operations and utilities
- **`openai_api.py`**: LLM API interface (user-implemented)
- **`evaluation_steps.py`**: Evaluation utilities

### Frontend (`frontend/src/`)

- **`App.jsx`**: Main application component with global state management
- **`SchemaTab.jsx`**: Interactive schema designer
- **`DatabaseTab.jsx`**: Database record management interface
- **`DatasetTab.jsx`**: Dataset synthesis and download
- **`AnalysisTab.jsx`**: SQL-NL alignment analysis
- **`SQLSubexpressionCorrespondence.jsx`**: Visual representation of SQL components

### Configuration Files

- **`manual_config.json`**: Manual probability configuration for SQL synthesis
- **`learned_config.json`**: Learned probability distribution from existing datasets
- **`spider_example_pool.json`**: Example pool for in-context learning

## 📖 Usage

### Web Interface

#### 1. Schema Design

- **Import Schema**: Drag and drop a JSON schema file
- **Edit Schema**: Add/remove tables and columns
- **Define Relationships**: Specify primary and foreign keys
- **Add Descriptions**: Document tables and columns for better NL generation

Schema format example:
```json
{
  "users": {
    "comment": "User information table",
    "columns": [
      {
        "field": "user_id",
        "type": "text",
        "isPrimary": true,
        "comment": "Unique user identifier"
      },
      {
        "field": "username",
        "type": "text",
        "comment": "User's login name"
      }
    ]
  }
}
```

#### 2. Record Synthesis

- Click "Generate Records" to create synthetic data
- Specify the number of records to generate
- Records respect foreign key constraints and data types
- Export records to JSON

#### 3. SQL Query Synthesis

- Configure query distribution (number of tables, columns, clauses)
- Generate individual queries or batch synthesis
- View step-by-step SQL decomposition
- Get suggested natural language descriptions
- Check alignment between SQL and NL

#### 4. Dataset Analysis

- Upload existing SQL query datasets
- View comprehensive statistics:
  - Keyword distribution
  - Query structure patterns
  - Clause complexity
  - Column and table usage
  - Query complexity metrics

### Script-Based Synthesis

For large-scale dataset generation without the UI:

```python
from server import auto_synthetic_data

synthetic_data = auto_synthetic_data(
    schema_path="backend/saved_frontend_schema.json",
    save_path="backend/output_data/synthetic_data.jsonl",
    config_path="backend/learned_config.json",
    synthesized_DB_records_path="backend/output_data/DB_records.json",
    example_path="backend/spider_example_pool.json",
    data_num=2000
)
```

**Parameters**:
- `schema_path`: Path to the database schema JSON file
- `save_path`: Output file path for synthetic data
- `config_path`: Configuration file for query distribution
- `synthesized_DB_records_path`: Path to save generated database records
- `example_path`: Path to example pool for in-context learning
- `data_num`: Number of SQL-NL pairs to generate

## ⚙️ Configuration

### Query Distribution Configuration

Adjust probabilities in `learned_config.json` or `manual_config.json`:

```json
{
  "sample_table_probs": [0.5, 0.3, 0.2],
  "sample_column_probs": [0.4, 0.3, 0.2, 0.1],
  "select_star_prob": 0.2,
  "where_clause_prob": 0.3,
  "group_by_clause_prob": 0.2,
  "order_by_clause_prob": 0.3,
  "having_clause_prob": 0.3,
  "limit_clause_count": 0.1
}
```

### Network Configuration

#### Change Backend Port

Edit `backend/server.py`:
```python
app.run(debug=True, host="0.0.0.0", port=YOUR_PORT)
```

#### Change Frontend Port

```bash
# macOS/Linux
PORT=4000 npm start

# Windows
set PORT=4000 && npm start
```

#### Deploy on Server

Replace `localhost` with your server IP in `frontend/src/App.jsx`:
```javascript
const ip = 'your.server.ip';  // or domain name
const port = 5001;
```

## 🔌 API Reference

### Key Endpoints

#### `POST /step_by_step_description`
Generate step-by-step explanation for a SQL query.

**Request**:
```json
{
  "sql": "SELECT name FROM users WHERE age > 18",
  "schema": {...}
}
```

**Response**:
```json
{
  "explanation_data": [...]
}
```

#### `POST /suggested_nl`
Get suggested natural language description for SQL.

**Request**:
```json
{
  "sql": "...",
  "schema": {...},
  "parsed_step_by_step_data": [...]
}
```

**Response**:
```json
{
  "nl_query": "What are the names of users older than 18?",
  "examples": [...]
}
```

#### `POST /check_alignment`
Check alignment between NL and SQL components.

**Request**:
```json
{
  "sql": "...",
  "nl": "...",
  "schema": {...},
  "parsed_step_by_step_data": [...]
}
```

**Response**:
```json
{
  "alignment_data": [...],
  "uncovered_substrings": [...]
}
```

#### `POST /synthesize_records`
Generate synthetic database records.

**Request**:
```json
{
  "schema": {...},
  "num": 100
}
```

**Response**:
```json
{
  "synthetic_records": {...}
}
```

#### `POST /synthetic_sql`
Generate a random SQL query.

**Request**:
```json
{
  "schema": {...},
  "records": {...}
}
```

**Response**:
```json
{
  "synthetic_sql": "SELECT ...",
  "config": {...}
}
```

#### `POST /analyze_dataset`
Analyze an uploaded SQL query dataset.

**Request**: Multipart form data with file upload

**Response**:
```json
{
  "totalQueries": 1000,
  "averageComplexity": 12.5,
  "keywordDistribution": {...},
  "structureDistribution": {...},
  ...
}
```

## 👥 User Study

The `user_study/` folder contains 166 Spider database schemas for evaluation purposes. These schemas can be directly imported into SQLsynth:

1. Navigate to the Schema Tab
2. Drag and drop any `.json` schema file from `user_study/spider_schemas/`
3. The schema will be automatically loaded

These schemas are useful for:
- Testing SQLsynth on diverse database structures
- Benchmarking against the Spider text-to-SQL dataset
- Conducting user studies on SQL-NL alignment

## 🛠️ Development

### Project Structure

```
SQLsynth_repo/
├── backend/
│   ├── server.py              # Main Flask server
│   ├── SQL_synthesizer.py     # Query synthesis engine
│   ├── SQL2NL_clean.py        # Rule-based SQL parser
│   ├── llm_analysis.py        # LLM prompts and analysis
│   ├── records_synthesizer.py # Record generation
│   ├── ICL_retriever.py       # Example retrieval
│   ├── db_handling.py         # Database utilities
│   ├── openai_api.py          # LLM API interface
│   ├── evaluation_steps.py    # Evaluation tools
│   ├── *_config.json          # Configuration files
│   ├── output_data/           # Generated datasets
│   └── temp_db/               # Temporary databases
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── SchemaTab.jsx      # Schema designer
│   │   ├── DatabaseTab.jsx    # Record management
│   │   ├── DatasetTab.jsx     # Dataset synthesis
│   │   └── AnalysisTab.jsx    # Analysis interface
│   ├── public/                # Static assets
│   └── package.json           # Dependencies
├── user_study/
│   └── spider_schemas/        # 166 Spider schemas
└── README.md
```

## 📝 Citation

If you use SQLsynth in your research, please cite:

```bibtex
@inproceedings{Tian_2025, series={IUI ’25},
   title={Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation},
   url={http://dx.doi.org/10.1145/3708359.3712083},
   DOI={10.1145/3708359.3712083},
   booktitle={Proceedings of the 30th International Conference on Intelligent User Interfaces},
   publisher={ACM},
   author={Tian, Yuan and Lee, Daniel and Wu, Fei and Mai, Tung and Qian, Kun and Sahai, Siddhartha and Zhang, Tianyi and Li, Yunyao},
   year={2025},
   month=mar, pages={1398–1425},
   collection={IUI ’25} }

```


## 🙏 Acknowledgments

- Adobe Property

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact by tian211@purdue.edu.


