"""
Utilities for lazy imports of optional dependencies.
"""
import importlib

class LazyLoader:
    """
    Lazily import a module only when its attributes are accessed.
    
    This allows optional dependencies to be imported only when actually used,
    rather than at module import time.
    
    Usage:
        # Instead of: import chromadb
        chromadb = LazyLoader("chromadb")
        
        # Then use chromadb as normal - it will only be imported when accessed
        # If the module is not installed, a helpful error message is shown
    """
    def __init__(self, module_name, package_name=None):
        """
        Initialize a lazy loader for a module.
        
        Args:
            module_name: The name of the module to import
            package_name: Optional package name for pip install instructions
                         (defaults to module_name if not provided)
        """
        self._module_name = module_name
        self._package_name = package_name or module_name
        self._module = None
    
    def __getattr__(self, name):
        """Called when an attribute is accessed."""
        if self._module is None:
            try:
                # Use importlib.import_module instead of __import__ for better handling of nested modules
                self._module = importlib.import_module(self._module_name)
            except ImportError:
                raise ImportError(
                    f"The '{self._module_name}' module is required but not installed. "
                    f"Please install it with: pip install {self._package_name} "
                    f"or pip install dsrag[{self._package_name}]"
                )
        
        # Try to get the attribute from the module
        try:
            return getattr(self._module, name)
        except AttributeError:
            # If the attribute is not found, it might be a nested module
            try:
                # Try to import the nested module
                nested_module = importlib.import_module(f"{self._module_name}.{name}")
                # Cache it on the module for future access
                setattr(self._module, name, nested_module)
                return nested_module
            except ImportError:
                # If that fails, re-raise the original AttributeError
                raise AttributeError(
                    f"Module '{self._module_name}' has no attribute '{name}'"
                ) 

# Create lazy loaders for commonly used optional dependencies
instructor = LazyLoader("instructor")
openai = LazyLoader("openai")
cohere = LazyLoader("cohere")
voyageai = LazyLoader("voyageai")
ollama = LazyLoader("ollama")
anthropic = LazyLoader("anthropic")
genai = LazyLoader("google.generativeai", "google-generativeai")
boto3 = LazyLoader("boto3")
faiss = LazyLoader("faiss", "faiss-cpu")
psycopg2 = LazyLoader("psycopg2", "psycopg2-binary")
pgvector = LazyLoader("pgvector")