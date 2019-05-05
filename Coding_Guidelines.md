# Naming - 

  All class names should follow "CamelHump" naming style
  All file name should follow "this_naming_style"
  All function name should follow "verySimilarToCamelHump" style
  
  ToDo - Naming is not being followed

# Documentation - 

  Every function should have inital comments in the following format
  
    1-3 Line description of the function
    :input:
      :param parameter1: Description of parameter1
      :param parameter2: Description of parameter2
                          Description of parameter2 continued
    :output:
      :param parameter1: Description of parameter1
      :param parameter2: Description of parameter2
                          Description of parameter2 continued
                          
  There should be a "requirements.txt" file which contains all the packages used as well as their version

# Pipeline - 

  All programs will be stored in the "src" folder
  All constants related to your program will be stored in "configs" directory following the yaml style of writing
  There is only one "main.py" file which will be used to drive all the other functions

# Experiment Folder - 

  All files of the current run should be stored in the "Exp" directory whose path will be given in the "config" yaml files.
  If the Experiment folder already exists, user should be given a choice to rename the older folder or to replace it.
  
  ToDo - Add this functionality

# Logger - 

  There should be a logger function which stores the time-stamp of each output in the file "Exp/log.txt". It should also print the output on the console screen.

# Saving Work - 

  The "work" should be saved such that worst case scenario only 5 mins of work is lost. 
  **Care should be taken not to store the "work" too often which would cause harddisk overload**
  The frequency can be changes in real time by changing the corresponding parameters in the yaml files in "configs" directory

# Misc - 

  All the other "things" which has not been mentioned in the pipeline can be kept in the folder misc
  
# DockerFile - 
  
  After every version a docker file should be generated of the entire model
