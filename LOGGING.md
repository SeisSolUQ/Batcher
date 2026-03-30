# Batcher Logging

## Overview
The batcher now includes simple logging to help debug crashes and track what's happening with your batches.

## Log File Location
Logs are written to `batcher.log` in the current working directory.

## Log Format
Each log entry follows this format:
```
YYYY-MM-DD HH:MM:SS | MESSAGE
```

## What Gets Logged

### 1. Request Received
Every time a request comes in:
```
Request received: config={"order": "3"}, parameters=[[0.1, 0.2]]
```

### 2. Batch Submitted
When a batch is full or timeout is reached and sent to the simulator:
```
Batch submitted: config={"order": "3"}, parameters=[[0.1, 0.2], [0.1, 0.2]], real_count=2
```
- Shows all parameters in the batch (including any padding)
- `real_count` shows how many are real submissions (excluding padding)

### 3. Output Received
When the simulator returns output:
```
Output received: config={"order": "3"}, parameters=[[0.1, 0.2], [0.1, 0.2]], output=[0.5, 0.5]
```

Or if it fails:
```
Output FAILED: config={"order": "3"}, parameters=[[0.1, 0.2]], error=Connection timeout
```

## Example Log

Here's a complete trace of one batch:
```
2026-03-30 11:02:20 | Request received: config={"order": "3"}, parameters=[[0.1, 0.2]]
2026-03-30 11:02:20 | Batch submitted: config={"order": "3"}, parameters=[[0.1, 0.2], [0.1, 0.2]], real_count=2
2026-03-30 11:02:20 | Output received: config={"order": "3"}, parameters=[[0.1, 0.2], [0.1, 0.2]], output=[0.5, 0.5]
```

## Debugging Tips

### View logs in real-time
```bash
tail -f batcher.log
```

### Find failed batches
```bash
grep "Output FAILED" batcher.log
```

### Count requests vs outputs
```bash
grep -c "Request received" batcher.log
grep -c "Output received" batcher.log
```

### Find a specific config
```bash
grep '"order": "3"' batcher.log
```

That's it! Simple and straightforward logging.

