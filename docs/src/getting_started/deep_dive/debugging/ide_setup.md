# Debugging our pipeline code
This section provides comprehensive guidance on debugging the Matrix pipeline and documentation system. Whether you're troubleshooting pipeline issues, setting up debugging environments, or working with documentation, you'll find the tools and techniques you need here.

## Debugging with VS Code

### Setup

1. Create or modify `.vscode/launch.json` in your project root:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "kedro run",
            "type": "debugpy",
            "request": "launch",
            "module": "kedro",
            "args": ["run"],
            "cwd": "${workspaceFolder}/pipelines/your_pipeline"
        }
    ]
}
```

### Common Debug Configurations 

Here are some useful debug configurations for Kedro:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Full Pipeline",
            "type": "debugpy",
            "request": "launch", 
            "module": "kedro",
            "args": ["run"],
            "cwd": "${workspaceFolder}/pipelines/your_pipeline"
        },
        {
            "name": "Run Specific Nodes",
            "type": "debugpy",
            "request": "launch",
            "module": "kedro",
            "args": ["run", "--nodes", "node1,node2"],
            "cwd": "${workspaceFolder}/pipelines/your_pipeline"
        },
        {
            "name": "Run with Different Env",
            "type": "debugpy", 
            "request": "launch",
            "module": "kedro",
            "args": ["run", "-e", "test"],
            "cwd": "${workspaceFolder}/pipelines/your_pipeline"
        }
    ]
}
```

### Using the Debugger

1. Set breakpoints by clicking to the left of the line numbers in your code
2. Select your debug configuration from the dropdown in the Debug view
3. Start debugging by pressing F5 or clicking the green play button

### Debug Variables and Watch

- Use the Variables pane to inspect local and global variables
- Add expressions to the Watch pane to monitor specific values
- Use the Debug Console to evaluate expressions in the current context

### Advanced Debugging

#### Node-specific Debugging

To debug specific nodes in your pipeline:

```json
{
    "name": "Debug Specific Node",
    "type": "debugpy",
    "request": "launch",
    "module": "kedro",
    "args": ["run", "--nodes", "target_node"],
    "cwd": "${workspaceFolder}/pipelines/your_pipeline"
}
```

#### Pipeline Slicing

Debug a slice of your pipeline:

```json
{
    "name": "Debug Pipeline Slice",
    "type": "debugpy",
    "request": "launch",
    "module": "kedro",
    "args": ["run", "--from-nodes", "start_node", "--to-nodes", "end_node"],
    "cwd": "${workspaceFolder}/pipelines/your_pipeline"
}
```

### Additional Resources

- [VS Code Python Debugging Documentation](https://code.visualstudio.com/docs/python/debugging)
- [Kedro Documentation on Debugging](https://kedro.readthedocs.io/en/stable/development/debugging.html)

### Tips and Tricks

1. Use conditional breakpoints by right-clicking on a breakpoint
2. Enable "Just My Code" to avoid stepping into library code
3. Use logpoints for non-breaking debugging messages
4. Utilize the "Debug Console" for interactive debugging

### Common Issues

1. **Missing debugpy**: Ensure debugpy is installed in your environment
   ```bash
   uv add debugpy --dev
   # or if you're not in the pipelines/matrix directory:
   cd pipelines/matrix && uv add debugpy --dev
   ```
2. **Wrong Working Directory**: Make sure the `cwd` in launch.json points to your pipeline directory
3. **Environment Issues**: Verify VS Code is using the correct Python interpreter with your Kedro environment 