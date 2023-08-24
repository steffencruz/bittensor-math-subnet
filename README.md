
<div align="center">

# **Bittensor Mathematics Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This repo contains a simple implementation of a bittensor subnet, and is based on the [bittensor subnet template](https://github.com/opentensor/bittensor-subnet-template.git).

For a more comprehensive introduction to subnets, see [bittensor subnet template](https://github.com/opentensor/bittensor-subnet-template.git)

# Introduction
**Natural language is emerging as a primary way with which we interact with software systems.** While many mathematics problems are already easily solvable with modern technology, it remains a challenge for computer systems to understand and answer such questions when they are framed in [natural language](https://bdtechtalks.com/2023/03/06/chatgpt-llm-mathematics/#:~:text=%E2%80%9C%5BLLMs%5D%20can't,seen%20in%20its%20training%20set.%E2%80%9D). 

While some recent advancements have been made [algorithmically](https://arxiv.org/pdf/2303.05398.pdf) and with [plugins](https://www.wolfram.com/wolfram-plugin-chatgpt/), it remains an important research area to explore.

This subnet is designed to incentivize the development of such systems by providing a mechanism for validators to reward miners for solving natural language mathematics problems. Furthermore, by using proximal policy optimization (PPO), miners can benefit further by fine-tuning their language models on the validation data.

Several notable features of this subnet are:
- **Unlimited validation data** - prevents lookup tables from being used by miners
- **Clean division of validation topics** - allows for easy expansion of the validation data and experiments on MoE system design
- **A straightforward scoring mechanism** - easy to understand and extend
- **Several baseline miners** - to get you started and to demonstrate the subnet's behvaiour



## Validation Data

Currently, the natural language mathematics problems are generated by a relatively simple [simulator](https://github.com/lukew3/mathgenerator) which makes use of templates, but this can be easily expanded to include more bespoke and complex problems. The simulator generates question and answer pairs, with optional metadata for analysis.

One obvious exploit of this network (and showcased in a the built-in `TemplateMiner`) is that, while effectively unlimited in diversiy, the template-based questions are in practice quite easy for miners to reverse-engineer in order to obtain an exact answer. This can be alleviated by using an LLM such as GPT-4 to rephrase the questions or even generate question-answer pairs, thereby removing the rigidity imposed by simple templates. This would make it much more difficult for miners to reverse-engineer the template or otherwise bypass the proof-of-work mechanism.

Another interesting area for future work is to employ **adaptive difficulty** in the validation process, which would ensure that miners are constantly forced to improve and that the subnet remains competitive. One potential way to do this would be to use GPT-4 to produce new generator functions which are more difficult to solve, and to use these new generators to produce validation data. Another improvement would be to make use of other open source databases of mathematics, including [exams](https://www.maths.cam.ac.uk/undergrad/pastpapers), [textbooks](https://openstax.org/subjects/math), and [mathematics encyclopedias](https://encyclopediaofmath.org/wiki/Main_Page) and [olypiads](https://www.imo-official.org/problems.aspx). Other recent developments in this area are [MathQA](https://arxiv.org/pdf/2103.03874.pdf), which is a dataset of 20,000 questions and answers which can be used to train a model to generate questions and answers, and [process supervision](https://arxiv.org/pdf/2305.20050.pdf) which encourages correct chain of thought reasoning and has a large supporting [dataset](https://github.com/openai/prm800k).
Most likely, a combination of these approaches would be used to produce a robust, non-repetitive and adaptive validation system.

## Validation Topics

The simulator currently samples problems from 140 topics within 7 distinct categories (see below). By default, all topics are used for validation but this is a hyperparameter which can be changed. Each topic uses a unique generator to produce question and answer pairs from templates. Because of this clean division of topics, it is easy to add new topics and generators to the subnet.

Moreover, it facilitates an interpretable routing mechanism to be trained which will route questions to the most capable miners, producing an efficient MoE system. This is a key part of the subnet, as it allows for the subnet to be scaled up to include more miners and topics.

![math_categories](https://github.com/steffencruz/bittensor-math-subnet/assets/6709103/294e50a9-4929-48f4-81fd-d5a23ffb90ea)


## Scoring Mechanism

Rewarding miner reponses is a key part of any subnet, but can be one of the most difficult parts to design in a robust and fair way. In this network, the scoring mechanism is extremely simple as a ground truth answer is provided for each question. The miner's response is then compared to the ground truth answer and a score is calculated based a configurable loss function (such as MSE or cross entropy). This score is then used to determine the reward for the miner.

A future improvement to this subnet would be to use a more complex scoring mechanism which takes into account the chain of thought used to arrive at the answer. This would allow for more complex problems to be solved, and would also allow for more complex scoring mechanisms to be used. However, with the increased complexity comes the increased risk of miners gaming the system, so this is a tradeoff which must be considered.

## Supported Miners

This subnet currently supports the following miners out of the box:

- `SimpleEvalMiner` - a miner which naively evaluates the question using the `eval` function (spoiler: it's really bad)
- `SimPyMiner` - a miner which uses the `sympy` library to attempt to answer the question (spoiler: it's also really bad)
- `OpenAIMiner` - a miner which uses the OpenAI API to generate responses (requires an OpenAI API key)
- `TemplateMiner` - a miner which uses the `mathgenerator` library to generate responses (spoiler: it's really good)

An interesting avenue for further work is for miners to develop plugin-like integrations, which would enable them to use more complex solvers such as [Wolfram Alpha](https://www.wolframalpha.com/), [SymPy Gamma](https://www.sympygamma.com/), and [Mathematica](https://www.wolfram.com/mathematica/). While this may apear to bypass the challenge presented by the subnet, it would actually be a good thing as it would create for competition at the level of parsing questions for these solvers, which would ultimately lead to more reliable and robust, and efficient solutions.

</div>

---

# Running the template
Before running the template you will need to attain a subnetwork on either Bittensor's main network, test network, or your own staging network. To create subnetworks on each of these subnets follow the instructions in files below:
- `docs/running_on_staging.md`
- `docs/running_on_testnet.md`
- `docs/running_on_mainnet.md`

If you are interested in running simulations or benchmarking experiments on this subnet, follow the instructions [below](#running-a-simulation).
</div>

---

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/opentensor/bittensor-math-subnet.git
cd bittensor-math-subnet
python -m pip install -r requirements.txt
python -m pip install -e .
```

</div>

---

Once you have installed this repo and attained your subnet, you can run the miner and validator with the following commands.
```bash
# To run the miner
python -m neurons/miner.py
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode

# To run the validator
python -m neurons/validator.py
    --netuid <your netuid> # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url> # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

## Running a Simulation
If you are interesting in running the subnet for educational and benchmarking purposes you can run the following:


*create and register a set of peers belonging to the same coldkey*:
```bash
python populate.py --n 10 --wallet.name <your wallet name> --wallet.hotkey <your hotkey>
```

*run the simulation*:
```bash
python simulate.py 
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --config <path to config file> # Contains the network configuration such as miners and validators
    --wandb # Log results to wandb (optional)
```
This will run a local experiment (or tracked experiment if `--wandb` is specified) and will log the results to the console and to wandb (if specified). The results will also be saved to a file in the `results` directory.

</div>

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
