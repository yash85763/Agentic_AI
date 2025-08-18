# Pydantic Implementation
from typing import Callable, List, Union, Any
from pydantic import BaseModel, Field, validator

class PydanticTool(BaseModel):
    """
    A Pydantic-based class representing a reusable piece of code (Tool).
    """
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="A textual description of what the tool does")
    func: Callable = Field(..., description="The function this tool wraps")
    arguments: List[tuple] = Field(..., description="A list of (arg_name, arg_type) tuples")
    outputs: Union[str, List[str]] = Field(..., description="The return type(s) of the wrapped function")
    
    class Config:
        # Allow arbitrary types (like Callable)
        arbitrary_types_allowed = True
        # Don't validate assignment after initialization
        validate_assignment = True
    
    @validator('arguments')
    def validate_arguments(cls, v):
        """Ensure arguments is a list of tuples with (name, type) format"""
        if not isinstance(v, list):
            raise ValueError("arguments must be a list")
        for arg in v:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("Each argument must be a tuple of (name, type)")
            if not isinstance(arg[0], str):
                raise ValueError("Argument name must be a string")
        return v
    
    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)


# DSPy Implementation
import dspy
from typing import Callable, List, Union, Any

class DSPyTool(dspy.Module):
    """
    A DSPy-based class representing a reusable piece of code (Tool).
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: List[tuple],
                 outputs: Union[str, List[str]]):
        super().__init__()
        
        # Validate inputs
        self._validate_arguments(arguments)
        
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs
    
    def _validate_arguments(self, arguments: List[tuple]):
        """Validate the arguments format"""
        if not isinstance(arguments, list):
            raise ValueError("arguments must be a list")
        for arg in arguments:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("Each argument must be a tuple of (name, type)")
            if not isinstance(arg[0], str):
                raise ValueError("Argument name must be a string")
    
    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """
        DSPy's forward method - invoke the underlying function.
        This is DSPy's convention for the main computation method.
        """
        return self.func(*args, **kwargs)
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Invoke the underlying function (callable) with provided arguments.
        Delegates to forward() following DSPy conventions.
        """
        return self.forward(*args, **kwargs)


# Example usage and comparison
if __name__ == "__main__":
    # Define a sample function to wrap
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together"""
        return x + y
    
    # Create tools using both implementations
    pydantic_tool = PydanticTool(
        name="add_tool",
        description="Adds two numbers together",
        func=add_numbers,
        arguments=[("x", "int"), ("y", "int")],
        outputs="int"
    )
    
    dspy_tool = DSPyTool(
        name="add_tool",
        description="Adds two numbers together", 
        func=add_numbers,
        arguments=[("x", "int"), ("y", "int")],
        outputs="int"
    )
    
    # Test both tools
    print("Pydantic Tool:")
    print(pydantic_tool.to_string())
    print(f"Result: {pydantic_tool(5, 3)}")
    print()
    
    print("DSPy Tool:")
    print(dspy_tool.to_string())
    print(f"Result: {dspy_tool(5, 3)}")
    print()
    
    # Show validation in action (Pydantic)
    try:
        invalid_tool = PydanticTool(
            name="invalid",
            description="Invalid tool",
            func=add_numbers,
            arguments=["invalid"],  # Should be list of tuples
            outputs="int"
        )
    except Exception as e:
        print(f"Pydantic validation error: {e}")
