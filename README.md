### A purely insect-inspired model facilitating autonomous navigation by incorporating __goal approaching__ and __collision avoidance__

#### 1. Performance
two demos of agent's navigating in static and dynamic environment:
![](demo1.gif) | ![](demo2.gif) 
---|---

#### 2. Files
```
project
│   README.md
|   demo1.gif
|   demo2.gif
│
└───simulations
|     models.py: implementation of the goal approaching - path integration model (Stone et.al 2017), the collision avoidance - LGMD model (Yue and Rind, 2006) etc.
|     agent.py: combine the PI and LGMD to have the autonomous navigation model
|     plotter.py: some functions for plotting
|     runner.py: run trails of simulations
|     navigation_run_with_visulization.py: run the simulation with dynamically updated animation
|     visualization: generate video of animation / display the results by stored data containing 
│   │
│   └───worlds:contains the data of the simulated 3D world consists of static/dynamic obstactles
│       └───3FoodS200_Random: randomly moving obstacles
│       └───3FoodS200_Trans: tanslationally moving obstacles
│   
└───results
    └───simulation_navigation
    └───simulation_navigation_lgmd_no_enhancement
    └───simulation_navigation_random_obs
    └───simulation_navigation_varied_contrast
```

### References


