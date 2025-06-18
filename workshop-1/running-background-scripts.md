## Running Training Scripts in the Background

When working on a remote GPU server, it’s important to run your training scripts in a way that they keep running even if you disconnect from your SSH session. There are several methods to achieve this:

---

### 1. Using `nohup`

`nohup` ("no hang up") allows your script to continue running after you log out.

**Basic Usage:**

```bash
nohup python your_training_script.py > output.log 2>&1 &
```

- `> output.log` saves the output to a file.
- `2>&1` redirects errors to the same file.
- `&` runs the process in the background.

**Check running jobs:**

```bash
jobs -l
```

**Find your process:**

```bash
ps aux | grep python
```

**Stop your script:**
Find the process ID (PID) and kill it:

```bash
kill
```

---

### 2. Using `screen`

`screen` lets you create detachable terminal sessions.

**Start a screen session:**

```bash
screen -S train_session
```

**Run your script:**

```bash
python your_training_script.py
```

**Detach from session:**  
Press `Ctrl+A`, then `D`.

**List sessions:**

```bash
screen -ls
```

**Reattach:**

```bash
screen -r train_session
```

---

### 3. Using `tmux`

`tmux` is another terminal multiplexer, similar to `screen`.

**Start a tmux session:**

```bash
tmux new -s train_session
```

**Run your script:**

```bash
python your_training_script.py
```

**Detach:**  
Press `Ctrl+B`, then `D`.

**List sessions:**

```bash
tmux ls
```

**Reattach:**

```bash
tmux attach -t train_session
```

---

### 4. Monitoring GPU Usage

To monitor GPU usage and ensure your script is running:

```bash
watch -n 1 nvidia-smi
```

---

## Summary Table

| Method | Detachable? | Easy to Monitor? | Recommended For        |
| ------ | ----------- | ---------------- | ---------------------- |
| nohup  | No          | Output log only  | Simple, one-off runs   |
| screen | Yes         | Yes              | Interactive sessions   |
| tmux   | Yes         | Yes              | Advanced, multitasking |

---

## Example Workflow

1. SSH into your server.
2. Start a `screen` or `tmux` session.
3. Activate your Python environment:
   ```bash
   conda activate myenv
   ```
4. Run your training script:
   ```bash
   python train.py
   ```
5. Detach safely and log out. Your script keeps running!

---

**Tip:**  
Always redirect output to a log file for long runs, so you can check progress or errors later.


