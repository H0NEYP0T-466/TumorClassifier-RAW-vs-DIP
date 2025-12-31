# Contributing to TumorClassifier-RAW-vs-DIP

First off, thank you for considering contributing to TumorClassifier-RAW-vs-DIP! It's people like you that make this project such a great tool for the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
  - [Git Commit Messages](#git-commit-messages)
  - [TypeScript Style Guide](#typescript-style-guide)
  - [Python Style Guide](#python-style-guide)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible using our [bug report template](https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP/issues/new?template=bug_report.yml).

**Good bug reports include:**

- A clear and descriptive title
- Detailed steps to reproduce the issue
- Expected behavior vs. actual behavior
- Screenshots or error logs (if applicable)
- Environment details (OS, Python version, Node.js version, browser)
- Any relevant code snippets

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, use our [feature request template](https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP/issues/new?template=feature_request.yml).

**Good enhancement suggestions include:**

- A clear and descriptive title
- Detailed description of the proposed feature
- Explanation of why this enhancement would be useful
- Possible implementation approaches
- Examples from other projects (if applicable)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our style guidelines

3. **Test your changes** thoroughly
   - Run frontend tests: `npm test`
   - Run linters: `npm run lint`
   - Build the project: `npm run build`
   - Test backend: Run all Python tests and verify API endpoints

4. **Commit your changes** with a clear commit message
   ```bash
   git commit -m "feat: add amazing feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request** using our PR template

## Development Setup

### Prerequisites

- Node.js (v18.0.0+)
- Python (v3.8+)
- Git

### Frontend Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/TumorClassifier-RAW-vs-DIP.git
cd TumorClassifier-RAW-vs-DIP

# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m App.main
```

### Running Both Simultaneously

Use two terminal windows:

**Terminal 1 (Frontend):**
```bash
npm run dev
```

**Terminal 2 (Backend):**
```bash
cd backend
python -m App.main
```

## Style Guidelines

### Git Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - A new feature
- `fix:` - A bug fix
- `docs:` - Documentation only changes
- `style:` - Code style changes (formatting, missing semi-colons, etc.)
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `build:` - Changes to build system or dependencies
- `ci:` - CI/CD configuration changes
- `chore:` - Other changes that don't modify src or test files
- `revert:` - Reverts a previous commit

**Examples:**
```
feat: add tumor probability confidence score
fix: resolve CORS issue in API endpoints
docs: update installation instructions
refactor: optimize image preprocessing pipeline
```

### TypeScript Style Guide

- Use **TypeScript** for all new frontend code
- Follow existing code formatting (we use ESLint)
- Use meaningful variable and function names
- Add JSDoc comments for complex functions
- Prefer functional components and hooks over class components
- Use proper TypeScript types (avoid `any` when possible)

**Before committing:**
```bash
npm run lint
```

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use **type hints** for function parameters and return values
- Add docstrings to all functions, classes, and modules
- Keep functions focused and single-purpose
- Use meaningful variable names

**Code example:**
```python
def preprocess_image(image: np.ndarray, return_steps: bool = False) -> dict:
    """
    Apply preprocessing pipeline to the input image.
    
    Args:
        image: Input grayscale image as numpy array
        return_steps: If True, return intermediate preprocessing steps
        
    Returns:
        Dictionary containing processed image and optional steps
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Frontend Testing

- Write unit tests for utility functions
- Test React components with user interactions
- Ensure all existing tests pass before submitting PR

```bash
npm test
```

### Backend Testing

- Write unit tests for preprocessing functions
- Test API endpoints with various inputs
- Include edge cases and error handling
- Test model prediction accuracy

```bash
# Run Python tests (if test suite exists)
pytest
```

### Manual Testing

Before submitting a PR, manually test:

1. **Frontend:**
   - Image upload functionality
   - Model comparison view
   - Responsive design on different screen sizes
   - Error handling for invalid inputs

2. **Backend:**
   - All API endpoints (`/health`, `/predict`, `/predict/compare`)
   - Error responses for missing models
   - CORS configuration
   - Image preprocessing pipeline

## Documentation

- Update the README.md if you change functionality
- Add comments for complex logic
- Update API documentation for endpoint changes
- Include inline documentation for new functions/classes

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Keep formatting consistent with existing docs
- Use proper Markdown syntax

## Questions?

Don't hesitate to ask questions! You can:

- Open an issue with the `question` label
- Reach out to the maintainers

---

Thank you for contributing to TumorClassifier-RAW-vs-DIP! 🎉
