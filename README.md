# BIAS LLM-as-a-Judge Evaluation System

This work is part of the Europe Horizon project BIAS funded by the European Commission, and has received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI).
All work from the BIAS Project: https://github.com/BFH-AMI/BIAS 

## Overview

This system evaluates LLM answers for bias, inclusivity, and stereotype-free language using Large Language Models (LLMs) as judges. It processes Excel files containing both original and adapted (improved) prompts, evaluates them across multiple criteria, and generates comprehensive statistical analyses and visualizations.

## Citation

If you use this evaluation system in your research, please cite: TODO add Pre-Print Link here.

### Key Features

- **Multi-Criteria Evaluation**: Evaluates prompts on three dimensions:
  - No Stereotypes (bias detection)
  - Suitability for Inclusive Hiring
  - Inclusive Language Usage
- **Multiple LLM Judges**: Supports Mistral and GPT models as evaluators
- **Multiple Runs**: Performs repeated evaluations (default: 3 runs) for reliability
- **Comprehensive Logging**: Detailed logs capture all LLM interactions and evaluations
- **Statistical Analysis**: Calculates medians, averages, and comparative statistics
- **Visualizations**: Generates comparison charts between original and adapted prompts
- **Excel Integration**: Direct Excel file processing with column-based scoring

## Architecture

The system implements a **multi-model pipeline** architecture where specialized LLM agents evaluate prompts sequentially across different criteria. Each agent has a specific evaluation role, and the system processes prompts through all criteria for comprehensive assessment.

```
Input Excel → LLM Judge → Evaluate 3 Criteria → Score Original & Adapted → Output Files
                ↓
         [No Stereotypes]
         [Inclusive Hiring]
         [Inclusive Language]
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

Install dependencies using pip:

```bash
pip install pandas openpyxl requests matplotlib numpy argparse
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Requirements File

Create a `requirements.txt` file with:

```
pandas>=2.0.0
openpyxl>=3.1.0
requests>=2.31.0
matplotlib>=3.7.0
numpy>=1.24.0
```

## Configuration

### API Keys

The system requires API keys for the LLM judges. Create text files in the same directory as `run_evaluation.py`:

1. **For Mistral**: Create `api_key_mistral.txt`
   ```
   your-mistral-api-key-here
   ```

2. **For OpenAI/GPT**: Create `api_key_openai.txt`
   ```
   your-openai-api-key-here
   ```

**Important**: Keep these files secure and add them to `.gitignore` to prevent accidental commits.

### Excel File Structure

Your input Excel file must have the following column structure:

| Column | Content | Used For |
|--------|---------|----------|
| G | Original Prompt | Input for evaluation |
| H | No Stereotypes (Original) | Output score |
| I | Suitable for Inclusive Hiring (Original) | Output score |
| J | Inclusive Language (Original) | Output score |
| M | Adapted Prompt | Input for evaluation |
| N | No Stereotypes (Adapted) | Output score |
| O | Suitable for Inclusive Hiring (Adapted) | Output score |
| P | Inclusive Language (Adapted) | Output score |

**Note**: Columns H, I, J, N, O, P should be empty in the input file - they will be populated with scores (1-10) by the evaluation system.

## Usage

### Basic Command

Evaluate an Excel file with default settings (3 runs, Mistral judge):

```bash
python run_evaluation.py --input your_file.xlsx
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Path to input Excel file |
| `--judge` | No | `mistral` | LLM judge to use (`mistral` or `gpt`) |
| `--runs` | No | `3` | Number of evaluation runs |
| `--analyze` | No | `False` | Generate statistical analysis and charts |

### Example Commands

**Standard evaluation with 3 runs:**
```bash
python run_evaluation.py --input prompts.xlsx --judge mistral
```

**Evaluation with statistical analysis:**
```bash
python run_evaluation.py --input prompts.xlsx --judge mistral --analyze
```

**5 runs with GPT-4 as judge:**
```bash
python run_evaluation.py --input prompts.xlsx --judge gpt --runs 5 --analyze
```

**Single run for testing:**
```bash
python run_evaluation.py --input prompts.xlsx --judge mistral --runs 1
```

## Output Files

The system generates multiple files for each run:

### 1. Evaluated Excel Files
- **Format**: `{input_filename}_Run{N}_evaluated.xlsx`
- **Example**: `prompts_Run1_evaluated.xlsx`
- **Content**: Original Excel with populated score columns (H, I, J, N, O, P)

### 2. Log Files
- **Format**: `{input_filename}_Run{N}_{timestamp}.log`
- **Example**: `prompts_Run1_20231221_143025.log`
- **Content**:
  - Run metadata (timestamp, input file, judge)
  - Complete prompt text for each evaluation
  - Full LLM questions sent to judge
  - Complete LLM responses
  - Extracted scores
  - Progress indicators
  - Statistical summaries

### 3. Visualization Charts (if `--analyze` is used)
- **Format**: `{input_filename}_Run{N}_comparison.png`
- **Example**: `prompts_Run1_comparison.png`
- **Content**: Side-by-side bar charts comparing Original vs Adapted prompts
  - Left chart: Median scores
  - Right chart: Average scores

### Example Output Structure

After running with 3 runs and analysis:
```
prompts_Run1_evaluated.xlsx
prompts_Run1_20231221_143025.log
prompts_Run1_comparison.png
prompts_Run2_evaluated.xlsx
prompts_Run2_20231221_144130.log
prompts_Run2_comparison.png
prompts_Run3_evaluated.xlsx
prompts_Run3_20231221_145235.log
prompts_Run3_comparison.png
```

## Evaluation Criteria

### 1. No Stereotypes
Evaluates whether the answer avoids societal stereotypes or biases.
- **Scale**: 1-4 (1=fails badly, 2=fails, 3=works ok, 4=works very well)
- **Higher scores are better**

### 2. Suitable for Inclusive Hiring
Assesses whether the answer promotes inclusive hiring practices.
- **Scale**: 1-4 (1=fails badly, 2=fails, 3=works ok, 4=works very well)
- **Higher scores are better**

### 3. Inclusive Language
Evaluates the use of inclusive language in the answer.
- **Scale**: 1-4 (1=fails badly, 2=fails, 3=works ok, 4=works very well)
- **Higher scores are better**

## Statistical Analysis

When using the `--analyze` flag, the system calculates:

- **Mean (Average)**: Average score across all prompts
- **Median**: Middle value when scores are sorted
- **Standard Deviation**: Measure of score variability
- **Min/Max**: Lowest and highest scores
- **Improvement Metrics**: Difference between Adapted and Original prompts

These statistics are logged and visualized in comparison charts.

## Troubleshooting

### API Key Issues

**Error**: `Judge 'mistral' is not configured`

**Solution**: 
1. Check that `api_key_mistral.txt` exists in the same directory
2. Verify the API key is valid and not expired
3. Ensure there are no extra spaces or newlines in the file

### Excel Column Errors

**Error**: Scores not appearing in output

**Solution**:
1. Verify your Excel has columns G and M with prompts
2. Ensure columns H, I, J, N, O, P exist (can be empty)
3. Check that the Excel file is not password-protected

### Missing Scores

**Error**: Some scores show as empty in output

**Cause**: LLM response didn't match expected format

**Solution**: 
1. Check the log file for the raw LLM response
2. The system tries multiple regex patterns to extract scores
3. Manual review may be needed for malformed responses

### Rate Limiting

**Issue**: API calls failing intermittently

**Solution**:
1. Add delays between requests (modify code if needed)
2. Use a higher-tier API plan
3. Run fewer rows per batch

## Performance Considerations

- **Time Estimate**: ~2-5 seconds per evaluation (6 evaluations per row)
- **For 10 rows**: ~2-5 minutes per run
- **For 50 rows**: ~10-25 minutes per run
- **API Costs**: Each row requires 6 LLM API calls (3 criteria × 2 prompts)

## Best Practices

1. **Test First**: Run with `--runs 1` on a small subset to verify setup
2. **Monitor Logs**: Check log files during long runs to catch issues early
3. **API Key Security**: Never commit API keys to version control
4. **Backup Original**: Keep a copy of your input Excel before running
5. **Multiple Runs**: Use 3+ runs for more reliable statistical results
6. **Analysis**: Always use `--analyze` for comprehensive insights
