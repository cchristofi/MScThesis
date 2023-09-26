# MSc Artificial Intelligence Thesis GitHub Repository

## Title: Teaching Robots: Uncovering User Preferences and Interaction Patterns

### Abstract:
This thesis digs into a deep exploration of human-robot collaboration, specifically focusing on the innate human ability to interact intuitively with robots and provide feedback. The primary objective of this research is to explore the intricate dynamics of human-robot interaction, with a particular focus on the robot learning aspect with the help of human teachers. The study delves into the intricate mechanics of user-driven teaching and interaction patterns, specifically examining the temporal dynamics of user engagement with teaching modalities: **Demonstrations** and **Evaluative Feedback**. 

![plot](https://github.com/cchristofi/MScThesis/blob/main/Images/Implementation.png)

Moreover, it investigates the potential impact of diverse demographic factors on these interaction patterns, seeking to decipher the nuanced influences that shape human-robot dynamics. Additionally, this research aims to explore the various factors that drive users' interactions with robots, delving into the intricate motivations and preferences that guide their engagement. To achieve these insights, the study employs a learning algorithm that leverages **Policy Shaping** and **Q-learning** Reinforcement Learning methods, which operate in the background as users interact with the robot. 

#### Policy Shaping:
$$
p_{c}(\alpha|s) = \frac{C^{\Delta _{s, \alpha }}}{C^{\Delta _{s, \alpha }}+ (1-C^{\Delta _{s, \alpha }})}
$$

#### Q-Learning:
$$
Q_{new}(s_{t}, \alpha_{t}) = Q(s_{t}, \alpha_{t}) + \alpha [r_{t} + \gamma \ max_{\alpha_{t+1}}Q(s_{t+1}, \alpha_{t+1}) - Q(s_{t}, \alpha_{t})]
$$

Through a series of experiments involving **58 participants** engaging with the robotic arm, the data we collected involving the teacher's behaviour during training undergoes meticulous analysis. Notably, while statistical significance is absent between the average time to intervene and demographic variables, a distinct pattern emerges: participants demonstrate a tendency to provide evaluative feedback earlier in their interactions compared to demonstrations. This exploratory analysis lays the foundation for future research attempts, suggesting the potential for more expansive investigations with larger participant groups. Ultimately, this study marks a significant step towards comprehending the complex dynamics of human-robot interaction, offering insights that lead the way for further exploration and understanding of how humans interact with robotic counterparts.

#### Environment
![plot](https://github.com/cchristofi/MScThesis/blob/main/Images/Task2.png)

## Installation
1. Install PyRep by following installation guide here: [PyRep: A toolkit for robot learning research.](https://github.com/stepjam/PyRep)
2. Install RLBench by following installation guide here: [RLBench: A large-scale benchmark and learning environment.](https://github.com/stepjam/RLBench)
3. Install panda-gym by following installation guide here: [panda-gym: Set of robotic environments based on PyBullet physics engine and gymnasium.](https://github.com/qgallouedec/panda-gym)
4. Install requirements file: 
```bash
pip install -r Code/requirements.txt
```

## Run
```bash
python3 Code/panda.py
```
