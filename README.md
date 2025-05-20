# Spring 2025 CS336 lectures

This repo contains the lecture materials for "Stanford CS336: Language modeling from scratch".

## Under implementation
I'm gradually execute the lectures and finish all the assignments in this repo.
### Update: Assignment 1 finished. May 20 ☕
The model with a BPE tokenizer is in [`assignment/assianment1-basics/cs336_basics`](https://github.com/CatManJr/spring2025-lectures/assignment/assianment1-basics/cs336_basics). I only run one experiment on my mac because I do not have the permission to use the H100.
The training result is acceptable with val loss at 1.71 but unluckily the pt file is to large (over 300MB) to be pushed to this repo.

#### Training Summary             
| Metric              | Value            |
| ------------------- | ---------------- |
| Total Training Time | 21209.29 seconds |
| Final Train Loss    | 1.7131           |
| Best Val Loss       | 1.7122           |
| Model Parameters    | 28.92M           |
| Total Iterations    | 10000            |
#### Generating sample
```Generated Sample Text:
 and the dog were playing with it. But then, Tim saw something strange. It was a toy outside! Tim picked up the toy and showed it to his mom. His mom was a little 
scared, but Tim had an idea.Tim showed her the toy and asked if it could play too. His mom agreed, and they played with the toy together. Tim was very happy and not 
fearful anymore. They all had a fun day at the park.<|endoftext|>Once 
```
There is still a issue in my `RoPE module`. I modified it to run training in defferent bath sizes, but then it cannot pass the pytest by Stanford. To pass the pytest, the training script can only run with `batch_size = 8` & `batch_size = 1`

## Non-executable (ppt/pdf) lectures

Located in `nonexecutable/`as PDFs

## Executable lectures

Located as `lecture_*.py` in the root directory

You can compile a lecture by running:

        python execute.py -m lecture_01

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

However, if you want to run it on the cluster, you can do:

        ./remote_execute.sh lecture_01

which copies the files to our slurm cluster, runs it there, and copies the
results back.  You have to setup the appropriate environment and tweak some
configs to make this work (these instructions are not complete).

### Frontend

If you need to tweak the Javascript:

Install (one-time):

        npm create vite@latest trace-viewer -- --template react
        cd trace-viewer
        npm install

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

        npm run dev

Deploy to the main website:

        cd trace-viewer
        npm run build
        git add dist/assets
        # then commit to the repo and it should show up on the website
