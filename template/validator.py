# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Steffen Cruz

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

import typing
import torch
import random
import math
import re
import mathgenerator
import bittensor as bt
from sympy.parsing.latex import parse_latex


class ArithmeticExpression(torch.nn.Module):
    def __init__(self, min_depth=1, max_depth=5):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self) -> str:
        """
        Makes a random math question.
        """
        return self._generate_expr(random.randint(self.min_depth, self.max_depth))

    def _generate_expr(self, n: int) -> str:
        if n == 0:
            return str(random.randint(1, 100))
        else:
            return '(' + self._generate_expr(n - 1) + random.choice(['+', '-', '*', '/']) + self._generate_expr(n - 1) + ')'

    def reward(self, prompt: str, response: float) -> float:
        """
        Reward the miner response to the eval request.

        Returns:
        - float: The reward value for the miner.
        """
        bt.logging.info(f"Prompt: {prompt}, Response: {response}")
        try:
            return math.exp( - (response - eval(prompt)) ** 2 )
        except:
            return 0 # handles division by zero and None responses



class TemplateExpression(torch.nn.Module):

    choices: dict = {
        'time_unit': ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'],
        'action': ['walk', 'run', 'swim', 'fly', 'drive', 'bike', 'sail', 'row', 'climb', 'crawl'],
        'dist_unit': ['meters', 'kilometers', 'miles', 'lightyears', 'parsecs', 'furlongs', 'fathoms', 'cubits', 'inches', 'feet'],
    }
    def __init__(self):
        # give and example natural language math question, framed as an exam question

        self.template = 'If it takes {random_number1} {time_unit1} to {action1} {random_number1} {dist_unit1}, how long does it take to {action2} {random_number2} {dist_unit2} in {time_unit}?'

    def forward(self) -> str:

        choices = dict(
            random_number1 = random.randint(1, 100),
            time_unit1 = random.choice(self.choices['time_unit']),
            action1 = random.choice(self.choices['action']),
            random_number2 = random.randint(1, 100),
            dist_unit1 = random.choice(self.choices['dist_unit']),
            action2 = random.choice(self.choices['action']),
            random_number3 = random.randint(1, 100),
            dist_unit2 = random.choice(self.choices['dist_unit']),
            time_unit = random.choice(self.choices['time_unit']),
        )
        query = self.template.format(**choices)
        correct_answer = 42
        return query, correct_answer

    def reward(self, prompt: str, response: float, correct_answer: float) -> float:
        """
        Reward the miner response to the template request.

        Returns:
        - float: The reward value for the miner.
        """
        bt.logging.info(f"Prompt: {prompt}, Response: {response}, Correct Answer: {correct_answer}")
        try:
            return 1 / min(1, (response - correct_answer) ** 2 )
        except:
            return 0



class MathGenerator(torch.nn.Module):
    
    # TODO: composite question - blend multiple topics together

    # each entry contains the following keys
    all_topics: list[dict] = [dict(zip(('index', 'title', 'generator', 'function_name', 'category'), t)) for t in mathgenerator.getGenList()]

    def __init__(self, topics: typing.Union[int, str, list[int], list[str], dict]=None):

        self._topics = self._select_topics(topics)

    def _select_topics(self, topics: typing.Union[int, str, list[int], list[str], dict], field: str='title') -> list[dict]:
        """Applies user selection to all topics to produce a subset of topics.

        Args:
            topics (typing.Union[int, str, list[int], list[str], dict], optional): User selection. Defaults to None.
            field (str, optional): Topic field to match against. Defaults to 'title'.

        Returns:
            list[dict]: Subset of topics.
        """

        if topics is None:
            topics = self.all_topics[:]
        elif isinstance(topics, int):
            topics = [self.all_topics[topics]]
        elif isinstance(topics, str): # interpret as regex
            topics = list(filter(lambda x: re.match(topics, x[field], re.IGNORECASE), self.all_topics))
        elif isinstance(topics, list) and isinstance(topics[0], str):
            topics = list(filter(lambda x: any(re.match(t, x[field], re.IGNORECASE) for t in topics), self.all_topics))
        elif isinstance(topics, list) and isinstance(topics[0], int):
            topics = [self.all_topics[i] for i in topics]
        elif isinstance(topics, dict):
            topics = [topic for key, val in topics.items() for topic in self._get_topics(val, field=key)]
        else:
            raise ValueError(f'Invalid type {type(topics)} for topics: {topics}')

        return topics

    def forward(self, latex=False, numeric_solution=True, return_topic=False, **kwargs) -> typing.Union[tuple[str, str], tuple[str, str, dict]]:
        """Generate an expression.

        Args:
            latex (bool, optional): Whether to return the expression in LaTeX format. Defaults to False.
            numeric_solution (bool, optional): Whether to return the solution as a number or a string. Defaults to True.
            return_topic (bool, optional): Whether to return the topic as well. Defaults to False.
            **kwargs: Keyword arguments to pass to the generator.

        Returns:
            tuple: (question, answer) or (question, answer, topic)
        """
        for i in range(101):
            
            topic = random.choice(self._topics)
            question, answer = topic['generator'](**kwargs)

            if not latex:
                question = parse_latex(question.replace('$', '').replace('=','').strip())
                answer = parse_latex(answer.replace('$', '').replace('=','').strip())

            if numeric_solution:
                try:
                    answer = float(answer)
                    break
                except ValueError:
                    continue

        if i == 100:
            raise RuntimeError('Could not generate a valid expression after 100 attempts.')

        if return_topic:
            return question, answer, topic

        return question, answer

