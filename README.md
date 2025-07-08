# Spring 2025 CS336 lectures

This repo contains the lecture materials for "Stanford CS336: Language modeling from scratch".

## Under implementation: course assignments
I'm gradually executing the lectures and finish all the assignments in this repo. `(last update: 07-08-2025)`
### Update: Assignment 1 finished. May 20 ☕
The Decoder-only model with RoPE, SwiGLU and a BPE tokenizer is in [`assignment/assianment1-basics/cs336_basics`](https://github.com/CatManJr/spring2025-notes-and-assignments/tree/main/assignments/assignment1-basics/cs336_basics). I only run one experiment on my mac because I do not have the permission to use the H100 provided by CS336 in Stanford.
The training result is acceptable with val loss at 1.71 but unluckily the weight file is to large (over 300MB) to be pushed to this repo.

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

### Update: Assignment 2 finished. May 31 ☕
I am not going to run all the experiments neither as I do not have any GPUs myself. But I'll again try to implement all the problems to pass the pytest. (May 24, 2025)
I imported my TransformerLM in [`assignment/assianment2-systems/cs336_basics`](https://github.com/CatManJr/spring2025-notes-and-assignments/tree/main/assignments/assignment2-systems/cs336-basics). And the tasks including Flash-Attention2 triton kernels are implemented and passed course pytest. Due to the lack of GPU, I haven't test my triton kernels.(May 31, 2025) I'm not sure if I would review and run them on GPUs in the future. All the code for the assignment is in [`assignment/assianment2-systems/cs336_systems`](https://github.com/CatManJr/spring2025-notes-and-assignments/tree/main/assignments/assignment2-systems/cs336_systems)
Triton related part finished: I test the triton kernels on A100-PCIE-40G and adopted the codes I used in `My Kernel Templates` in this repo too.(June 7)

### Update: Assignment 3 finished. July 8 ☕
This assignment provides guidance on how to use scaling laws to estimate the performance of larger models based on their lighter version. I don't have the admission of using Stanford training api. So I simply simulated the training process. Please turn to [`assignment/assignment3-scaling/cs336_scaling`](https://github.com/CatManJr/spring2025-notes-and-assignments/tree/main/assignments/assignment3-scaling/cs336_scaling) for more details.

### Update: Assignment 4 finished. July 8 ☕
I implemented the required APIs that aim to construct an acceptable dataset from unfiltered web content. All tests passed. As I don't want to pollute the official leaderboard, I implemented a similar scoring system running on macOS. The scoring results are as below:

`SUMMARY:`
Filtered data: filtered_data/combined_filtered.txt (0.0 MB)  
Tokenized data: results/tokenized_data.bin (0.0 MB)  
Best validation loss: 4.4334  
Best validation perplexity: 84.21  
Data quality score: 0.089  

The performance was very poor because I used a very sketchy piece of content.

### Update: Assignment 5 started. July 8 ☕
Yeah! 3 assignments in a row! Catman is on fire mate! :fire:

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
