## Food Stamp Program Simulator

Welcome! This is a simulator that gives the estimated impacts of a custom defined food stamps program on the local labor market as compared to the baseline program currently in place.

This simulator is part of an analysis of a potential implementation of SNAP in Puerto Rico and thus, works best for estimating program impacts in Puerto Rico. 

However, as an extension, you are able to analyze food stamp programs in any U.S. state. Note that the current implementation does not take into account local unemployment and food stamp accessibility conditions for locations other than Puerto Rico, so for improved accuracy consider adding these variables into the program for your desired region. More information on this is available in the Extension to U.S. States section.


## Usage

Follow the following steps to run the simulation:

1. Download this folder.

2. Create a python virtual environment (venv) within the folder.

3. In your terminal, set the directory to the folder and activate the virtual environment with the following code (replacing the beginning of the path to match the location of the folder on your computer):

```bash
cd Downloads
cd FSProgram_Simulator
& c:\[Path to your downloads]\FSProgram_Simulator\.venv\Scripts\Activate.ps1
```

4. Install packages if needed
```bash
pip install requests
pip install statsmodels scipy sci-learn
pip install statsmodels scipy scikit-learn
```

5. Run the Python script with the following code:

```bash
python Simulator.py
```

6. The simulator will prompt you to enter a variety of inputs to set up your custom food stamps program. Follow the displayed instructions to do so. If you need more information, consult the inputs section below.

7. The results of the simulation will be displayed.

For a demo of this process, refer to the SimulatorDemo video in this folder. In this video, two iterations of the simulation are run: one for Puerto Rico and one for California.


## Inputs

The simulator will request you to provide 4 inputs:

1. Year for the estimation: 

You must choose either 2023 or 2024 and enter this value as a number. This choice refers to the year during which the data being used for estimation was collected. 


2. U.S. state for estimations:

You must choose one of the 51 U.S. states or Puerto Rico. This choice refers to the location for which the food stamp program will be simulated. If you choose one of the U.S. states, the baseline food stamp program will be SNAP. If you choose Puerto Rico, the baseline food stamp program will be NAP. You must enter your location selection as the two letter (capitalized) abbreviation for that state or PR for Puerto Rico.

    2.1. US Census API Key:
    
    If you choose a location other than Puerto Rico, you will need to have a US Census API key in order to fetch the necessary data. If you do not already have this key, you can requst one at https://api.census.gov/data/key_signup.html. You must enter your US Census API key in its exact form for the simulation to work.


3. Maximum monthly benefit schedule for your food stamps program

You must enter 18 numerical values seperated by commas and with no spaces. The maximum benefit schedule refers to the maximum value of benefits a household can receive monthly given their size. For example, the SNAP maximum benefit schedule is: 291,535,766,973,1155,1386,1532,1751,1970,2189,2408,2627,2846,3065,3284,3503,3722,3941 where the value 291 means that a household with a size of 1 can receive a maximum of $291 a month in SNAP benefits and the value 3941 means that a household with size 18 can receive a maximum of $3941 a month in SNAP benefits.


4. Monthly income eligbility cutoff schedule for your food stamps program

You must enter 18 numerical values seperated by commas and with no spaces. The income eligibility cutoff schedule refers to the maximum income a household can have in order to be eligible to receive food stamps under the program. For example, the SNAP income eligibility cutoff schedule is: 1215,1644,2072,2500,2929,3357,3785,4214,4643,5072,5501,5930,6359,6788,7217,7646,8075,8504 where the value 1215 means that a household with size 1 must earn at most $1215 per month in order to qualify for SNAP and the value 8504 means that a household with size 18 must earn at most $8504 per month in order to quality for SNAP.


If you enter an invalid value for any of the four inputs, the simulator will prompt you to try again and will only continue when you enter a qualifying input. 

If you accidentaly enter an undesired input and/or you would like to start the simulator over again, you may press Ctrl+C to quit the simulation or wait for the simulation to complete and run it again.


## Extension to U.S. States

There are three variables that the simulator excludes from its analysis for U.S. states. These variables are as follows:

1. Unemployment

The unemployment rate for the individual's PUMA region. Either for 2023 or 2024 depending on which year was chosen for analysis.

This variable was used in the employment regression.


2. Food Stamp Availability

Ratio of the number of businesses authorized to accept the baseline program food stamps in the individual's PUMA region over the population count in the individual's PUMA region. Either for 2023 or 2024 depending on which year was chosen for analysis.

This variable was used in the benefit take up regression.


3. Food Stamp Lag Receipt

Number of individuals that received the baseline program food stamps in the individual's PUMA region. Either for 2022 if 2023 was choosen or for 2023 if 2024 was choosen for analysis.

This variable was used in the benefit take up regression.


## Interpretation oF Results

It is important to remember the simulation is an estimation and its outputs should be interpreted as a suggestion of the relative direction and magnitude with which a particular program will impact labor market outcomes.

The simulation provides four categories of outputs: Employment, Weekly Hours Worked, Benefit Take Up, and Fiscal Effects. For the first three categories, the simulation provides the significance level of the Simulated Benefit coefficient, the average elasticity, the average change in the measure, and the measure under both the baseline and new program. The fiscal effects provides the average benefit values and overall costs for both the baseline and new program. 

The significance level of the Simulated Benefit coefficient is a good way to tell how informative the result of the simulation is since it tells us how much the target variable (i.e. employment, hours worked, or take up) is varies with the size of food stamps benefits received. Thus, the lower the significance level, the better the simulation will be in producing an accurate estimate . 

If the input combination that you choose produces an insignificant Simulated Benefit coefficient for any of the target variables, consider adding additional variables which capture local labor market or food stamp conditions into your regression.
