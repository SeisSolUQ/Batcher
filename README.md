# Batcher

Batcher is a high-performance proxy service designed to optimize model evaluation requests by aggregating them into batches before forwarding them to a backend model. It is built on top of the [UM-Bridge](https://um-bridge-benchmarks.github.io/) protocol.

This service is particularly useful when the backend model (e.g., a simulation) benefits significantly from batched inputs (vectorization) but the client sends requests sequentially or in small groups.

## Features

*   **Request Batching**: Accumulates incoming requests into efficient batches.
*   **Dual Batch Configuration**: Supports two distinct batch sizes based on the request configuration (`order`).
*   **Concurrency**: Thread-safe handling of multiple simultaneous client connections.
*   **Dynamic Padding**: Automatically pads incomplete batches with the last valid sample when the timeout is reached, ensuring consistent batch shapes for the backend.
*   **Timeout Management**: Configurable timeout prevents requests from stalling indefinitely if a batch doesn't fill up.
*   **Multi-dimensional Input**: Fully supports input vectors of arbitrary dimensions.
*   **UM-Bridge Integration**: Seamlessly integrates as a UM-Bridge model, making it compatible with any UM-Bridge client.

## Requirements

*   Python 3.x
*   [UM-Bridge](https://pypi.org/project/umbridge/)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/SeisSolUQ/Batcher.git
    cd Batcher
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Start the Batcher service using the following command:

```bash
python batcher.py <backend_url> <model_name> <batch_size_1> <batch_size_2> <port> <timeout>
```

### Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `backend_url` | The URL where the backend model is running. | `http://localhost:4243` |
| `model_name` | The specific model name on the backend to forward requests to. | `forward` |
| `batch_size_1` | The default batch size to use. | `8` |
| `batch_size_2` | The batch size to use when `config["order"] == "4"`. | `2` |
| `port` | The port for this Batcher service to listen on. | `4242` |
| `timeout` | Maximum wait time (in seconds) for a batch to fill. | `5` |

### Example

To start the batcher listening on port `4242`, forwarding to a model named `forward` running at `http://localhost:4243`, with a default batch size of 16 (and 4 for specific high-order requests), and a timeout of 1 second:

```bash
python batcher.py http://localhost:4243 forward 16 4 4242 1.0
```

## How It Works

1.  **Receive**: The Batcher receives a request via the UM-Bridge protocol.
2.  **Queue**: The request is added to a pending batch associated with its configuration (specifically the `order` parameter).
3.  **Wait**: The request thread waits until either:
    *   The batch becomes full (reaches `batch_size`).
    *   The `timeout` expires.
4.  **Process**:
    *   If the batch is full, it is sent to the backend immediately.
    *   If the timeout expires and the batch is not empty, the batch is **padded** with copies of the last sample to match the required batch size, then sent.
5.  **Respond**: The backend's response is unpacked, and the specific result for the original request is returned to the client.

## Testing

The repository includes a suite of regression tests to ensure stability and correctness:

*   **Concurrency**: `test_concurrency.py` verifies thread safety under high load.
*   **Busy Waiting**: `test_busy_waiting.py` ensures the service waits efficiently without consuming excessive CPU.
*   **Padding**: `test_padding.py` checks that incomplete batches are correctly padded.
*   **Multi-dimensional**: `test_multidim.py` confirms support for vector inputs.

Run the tests using:

```bash
python3 test_concurrency.py
python3 test_busy_waiting.py
python3 test_padding.py
python3 test_multidim.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

