# In-Hospital Mortality
A python package for deployment of a custom model in Epic's Cloud Services environment. Predicts likelihood of death within 24 hours for admitted adult patients with a current patient class of Inpatient or Emergency.

Associated Records: `R HDA 100060`  

---
## How to...
#### Launch Slate Container
1. Enter the container: `docker start -i slate`  
2. Navigate to project workspace: `cd workspace/InHospitalMortality/deployment`
3. Exit Slate: `Ctrl` + `D`

#### Install Dependencies
1. In Slate, navigate to `pip_packages`: `cd workspace/InHospitalMortality/deployment/pip_packages`
2. Install or upgrade package versions: `dsutils install package==version` 

#### Create sample Payload
1. In Slate, navigate to project workspace: `cd workspace/InHospitalMortality/deployment`
2. Convert `train_data.tsv` to payload format: `dsutils make-ondemand-payload`  
*There's a new `--from-RW` flag for `dsutils make-ondemand-payload` that makes it possible to use Reporting Workbench reports with html/Report assistance for retrieving back values.*

#### Download Model, etc.
1. In the main project directory, update `MODEL_NAME`, `EXPERIMENT_ID`, and `RUN_ID` in `get_artifacts.py` for the MLflow repository where the model is saved.
2. Run `get_artifacts.py`.

#### Test Predict method
1. In Slate, navigate to project workspace: `cd workspace/InHospitalMortality/deployment`
2. Run test on sample payload: `dsutils ondemand`

#### Package for Deployment
1. Increment version in `definition.json`.
2. In Slate, navigate to project workspace: `cd workspace/InHospitalMortality/deployment`
3. Zip the model directory for upload into Hyperspace: `dsutils archive`  
*This command ignores anything in an .archiveignore file, so you can upload the zip file to Hyperspace without including any auxiliary files you use for testing or validation that you don't want sent to the cloud.*

--- 

