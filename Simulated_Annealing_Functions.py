import numpy as np
import random

def Create_Random_Group_Structure(num_fine_group_meshes, num_boundaries):

    # This method returns a random group structure.  In this case, that just means that it returns a vector of length num_fine_group_meshes that is mostly populated with 0s but has num_boundaries-many 1s randomly distributed within it.

    # Initialize structure without any boundaries:
    random_group_structure = []
    for i in range(num_fine_group_meshes):
        random_group_structure.append(0)

    for i in range(num_boundaries):

        # Select random location for the boundary:
        boundary_location = random.randrange(0,num_fine_group_meshes)

        if random_group_structure[boundary_location] == 1: # If there already exists a boundary at this location

                # Randomly section new locations until we find one that isn't occupied:
                while random_group_structure[boundary_location] == 1:
                    boundary_location = random.randrange(0,num_fine_group_meshes)

        # Add in the boundary at that location:
        random_group_structure[boundary_location] = 1

    # Return the random group structure:
    return random_group_structure
        


def Create_Neighboring_Group_Structure(initial_group_structure,jump_size=1):

    # This method takes in an initial group structure (as a vector of 1 and 0s) and returns a perturbed group structure (as a vector of 1s and 0s) where one of the boundaries has been moved up or down by one location on the fine grid.

    # Randomly choose which of the boundaries is going to be perturbed
    perturbed_boundary_number = random.randrange(1,50)

    # Set up the return object:
    perturbed_group_structure = initial_group_structure

    # Iterate through the fine grid until the desired boundary is reached:
    boundary_counter = 0
    for i in range(len(initial_group_structure)):

        if initial_group_structure[i] == 1: # If a boundary is encountered
            boundary_counter = boundary_counter + 1
            
            if boundary_counter == perturbed_boundary_number: # If this is boundary that will be perturbed
                
                # Randomly choose whether to move the boundary up or down:
                new_boundary_location = i + random.choice([-1*jump_size,jump_size]) 
                
                if (new_boundary_location < 0) or (new_boundary_location > (len(initial_group_structure)-1)) or (initial_group_structure[new_boundary_location] == 1): # If there already exists a boundary at that location or the new location is out of range

                    # Recursively call the function again and hope that a non-troublesome boundary location is chosen next time:
                    return Create_Neighboring_Group_Structure(initial_group_structure)

                else:

                    # Implement the perturbation and return the perturbed group structure:
                    perturbed_group_structure[i] = 0
                    perturbed_group_structure[new_boundary_location] = 1

                    return perturbed_group_structure


def Decide_Acceptance(original_error, new_error, temperature):

    # This method decides whether or not a perturbed group structure should be accepted based on the stochastic simulated annealing acceptance rule.

    if new_error < original_error:

        return True

    else: 

        accept_probaility = np.exp(-1*abs(new_error - original_error)/temperature)
        random_number = random.uniform(0, 1) 

        return (random_number < accept_probaility)


def Toy_Evaluation_Function(input_group_structure):

    # This function just returns the standard deviation of the vector of distances between the boundaries (i.e. the function is minimized when the boundaries are equally spaced along the fine grid)

    distances = []
    distance_counter = 0

    for i in range(len(input_group_structure)):

        if input_group_structure[i] == 1:
            distances.append(distance_counter)
            distance_counter = 0
        else:
            distance_counter = distance_counter + 1

    return np.std(distances)
