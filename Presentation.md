# Building a society simulator
## Implementing "War, Space, and the Evolution of Old World Complex Societies"

- paper by Peter turchin
- Was exposed to it via the /r/proceduralgeneration subreddit
- It contains a model that describes the rise and fall of large scale societies in the time period of 1500BCE to 1500CE
- This time range is divided into 3 eras: 
    - 1500BCE - 500BCE
    - 500BCE - 500CE
    - 500CE - 1500CE

### Challenges

- Getting datasets to run the model on
    - agricultural extent
        - Effective temperature
        - Desert areas
        - rivers
    - steppe locations
        - eventually had to mask to just NE hemisphere
    - all of these I had to work with GIS data
        - had to learn from scratch
        - did this slowly over a few months, then intensely over a 2-week period
    
    
- Manipulating the datasets into a usable format
    - Ocean/not ocean
    - Aggregating data into a grid
    - Aggregating partial data
- Implementing the model
    - model rules
    - model data
- Missing information in the paper
    - Suplemental information needed for information on datasets needed
    - Supplemental video shows preconstructed land bridges
    - Supplemental info says there was an optimized set of values
        - Only initial tested values are stated
        - Some tables showing variance with scaling these values
        - What is tau_2?
    - steppe land not shown as agriculturally viable, but not mentioned anywhere
    
- Tweaks to the model
    - littoral distance per age
        - needed due to performance - constant changes required recomputing data
    - simulate all cells
        - original model only simulates agricultural cells
        - I just make it so that 
        - I do not have extra agricultural bridges for diffusion
            - see supplemental video
    - add miltech transfer on invasion
- Things model still does not get right
    - Australia
    - Amazon rainforest
    - caucuses
    


# general outline
- Title
- About me
    - make me human
    - graduated, etc.
- problem
    - paper
    - What it does
- discuss challenges
    - data
    - transformation of data
- How I solved them
- Further questions


# Ideas
- for littoral cells:
    - basically do a divide and conquer strategy to calculate the cells
    - for each cell, calculate littoral neighbors
    - use those neighbors to calculate the next set of neighbors
        - just concat them all and make a set
        - cache them all in a collection per tile and only save the edges, then you can just search the edges incrementally
        
- Add probabilities to invasion direction
    - more likely to invade agricultural cells
    - Invasion gradient
- use lower boundary of cell for Effective Temperature aggregation
    - might allow for more stuff in each cell
- Use PET/aridity index to get a better estimate of agriculturally viable areas

        


















