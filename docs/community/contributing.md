# Contributing to dsRAG

We welcome contributions from the community! Whether it's fixing bugs, improving documentation, or proposing new features, your contributions make dsRAG better for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/dsRAG.git
   cd dsRAG
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Making Changes

1. Make your changes in your feature branch
2. Write or update tests as needed
3. Update documentation to reflect your changes
4. Run the test suite to ensure everything works:
   ```bash
   python -m unittest discover
   ```

## Code Style

We follow standard Python coding conventions:
- Use [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use meaningful variable and function names
- Write docstrings for functions and classes
- Keep functions focused and concise
- Add comments for complex logic

## Documentation

When adding or modifying features:
- Update docstrings for any modified functions/classes
- Update relevant documentation in the `docs/` directory
- Add examples if appropriate
- Ensure documentation builds without errors:
  ```bash
  mkdocs serve
  ```

## Testing

For new features or bug fixes:
- Add appropriate unit tests in the `tests/` directory
- Tests should subclass `unittest.TestCase`
- Follow the existing test file naming pattern: `test_*.py`
- Update existing tests if needed
- Ensure all tests pass locally
- Maintain or improve code coverage

Example test structure:
```python
import unittest
from dsrag import YourModule

class TestYourFeature(unittest.TestCase):
    def setUp(self):
        # Set up any test fixtures
        pass

    def test_your_feature(self):
        # Test your new functionality
        result = YourModule.your_function()
        self.assertEqual(result, expected_value)

    def tearDown(self):
        # Clean up after tests
        pass
```

## Submitting Changes

1. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request:
   - Go to the dsRAG repository on GitHub
   - Click "Pull Request"
   - Select your feature branch
   - Describe your changes and their purpose
   - Reference any related issues

## Pull Request Guidelines

Your PR should:
- Focus on a single feature or fix
- Include appropriate tests
- Update relevant documentation
- Follow the code style guidelines
- Include a clear description of the changes
- Reference any related issues

## Getting Help

If you need help with your contribution:
- Join our [Discord](https://discord.gg/NTUVX9DmQ3)
- Ask questions in the PR
- Tag maintainers if you need specific guidance

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project maintainers' guidance

Thank you for contributing to dsRAG! 