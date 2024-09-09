


def few_shot_prompting():
    text = """
    An insurance claim is a formal request made by a policyholder to their insurance company for 
    coverage or compensation for a covered loss or policy event. The process involves gathering and processing various details, 
    including the policyholder's information, such as policy number, type, coverage details, and demographics.
    Claim details include the claim number, date of loss, type of loss, location, description, estimated and actual loss amounts, 
    status, and settlement date. Financial information encompasses the premium amount, payment history, claim history, deductible, 
    and reimbursement amounts. For auto claims, specific details like the vehicle's VIN, make, model, year, value, and damage details are required. 
    In the case of home or property claims, information about the property address, type, value, and damage details is necessary. 
    Health claims involve patient information, diagnosis codes, treatment details, provider information, and payment details. 
    Fraud indicators are identified through potential red flags, anomalies, and suspicious claim patterns. 
    This comprehensive approach is used to evaluate the claim, determine coverage, and decide on the appropriate settlement.
    """
    prompt = f"""
    Create a detailed description of an insurance claim involving a specific incident type, 
    based on the structured information provided below. Ensure that the description covers 
    all relevant aspects of the claim, including policyholder details, claim specifics, 
    financial information, and any other pertinent data. The incident type should be clearly 
    defined and integrated into the narrative, illustrating how it impacts the claim process and outcome. Along with it generate a refernce based on customer name and customer ID
    ```{text}```
    """
    return prompt


def zero_shot_prompting():
    prompt = f"""
    Create a detailed description of an insurance claim involving a specific incident type. The description should comprehensively 
    cover all aspects of the claim, including policyholder details, claim specifics, financial information, and any other pertinent data. 
    Clearly define the incident type and explain how it affects the claim process and outcome. Ensure the narrative 
    is complete and informative, highlighting how each piece of information contributes to the overall evaluation and settlement of the claim.Along with it generate a refernce based on customer name and customer ID 
    """
    return prompt


def structured_prompting():
    structured_info = {
    "Policyholder Information": {
        "Policy Number": "123456789",
        "Policy Type": "Auto",
        "Coverage Details": {
        "Limits": "$100,000",
        "Deductibles": "$500"
        },
        "Demographics": {
        "Age": 45,
        "Gender": "Male",
        "Location": "New York, NY"
        }
    },
    "Claim Details": {
        "Claim Number": "987654321",
        "Date of Loss": "2024-08-01",
        "Loss Type": "Collision",
        "Loss Location": "Manhattan, NY",
        "Loss Description": "Rear-end collision at a traffic light",
        "Estimated Loss Amount": "$10,000",
        "Actual Loss Amount": "$9,500",
        "Claim Status": "Open",
        "Claim Settlement Date": "Pending"
    },
    "Financial Information": {
        "Premium Amount": "$1,200 annually",
        "Payment History": "Up to date",
        "Claim History": "No previous claims",
        "Deductible Amount": "$500",
        "Reimbursement Amount": "Pending"
    },
    "Specific Incident Information": {
        "Vehicle Information": {
        "VIN": "1HGCM82633A004352",
        "Make": "Honda",
        "Model": "Accord",
        "Year": 2020,
        "Vehicle Value": "$20,000",
        "Damage Details": "Rear bumper and trunk damage"
        }
    },
    "Fraud Indicators": {
        "Red Flags": "None identified",
        "Suspicious Patterns": "None identified"
    },
    "Additional Considerations": {
        "Data Quality": "Verified and consistent",
        "Data Privacy": "Compliant with GDPR",
        "Balance of Data": "Balanced distribution of claim types",
        "Synthetic Data": "No synthetic data used"
    }
    }
    prompt = f"""
    Create a detailed description of an insurance claim involving a collision, based on the structured information provided below. 
    Clearly define the incident type and explain how it impacts the claim process and outcome. The narrative 
    should be complete and informative, highlighting how each piece of information contributes to the overall evaluation and settlement of the claim. Along with it generate a refernce based on customer name and customer ID
    ```{structured_info}```
    """

    return prompt


def detailed_prompting():

    prompt = f"""
    Policyholder Information:

    Start by describing the policyholder's information, including their policy number, policy type, coverage limits, deductibles, and demographics such as age, gender, and location.
    Claim Details:

    Next, provide details about the claim itself. Include the claim number, the date of loss, the type of loss (e.g., collision, fire, theft), the location where the loss occurred, and a brief description of the incident. Mention both the estimated and actual loss amounts, as well as the current status of the claim (e.g., open, closed, pending) and the settlement date if applicable.
    Financial Information:

    Outline the financial aspects of the claim. This should include the policyholder's premium amount, their payment history, and any previous claims they’ve made. Specify the deductible amount associated with the policy and any reimbursement amounts that have been processed or are pending.
    Specific Incident Information:

    If the claim involves a specific asset like a vehicle or property, describe this in detail. For an auto claim, include the vehicle’s VIN, make, model, year, value, and a summary of the damages. For a property claim, provide details such as the property address, type (house, apartment, etc.), value, and extent of the damage.
    Fraud Indicators:

    Assess the claim for any potential red flags or anomalies that might indicate fraud. Mention any patterns of suspicious claims or behaviors that were observed during the evaluation.
    Additional Considerations:

    Finally, discuss any additional considerations that were taken into account during the claim processing. This might include the quality and consistency of the data, adherence to data privacy and security regulations, and whether the data was balanced or if synthetic data was used.
    Conclusion:

    Summarize how the incident type impacts the claim process and outcome. Ensure the narrative is cohesive and informative, highlighting how each piece of information contributes to the overall evaluation and settlement of the claim and do write it as a story. Along with it generate a refernce based on customer name and customer ID

    """

    return prompt

def policy_summarization(prompt_name):
    prompt = f"""
Based on the detailed insurance claim description provided below, create a concise summary that highlights the most important aspects of the claim. 
The summary should include the incident type, key claim details (such as dates, amounts, and status), relevant policyholder information, 
and the final outcome or resolution of the claim. The goal is to provide a clear and quick overview of the claim in a few sentences Along with it generate a refernce based on customer name and customer ID.
```{prompt_name}```
"""
    return prompt

def claim_processing(prompt_name):
    prompt = f"""
Based on the detailed insurance claim description provided below, outline the steps involved in processing this claim. 
Include key actions such as verifying the policyholder's information, assessing the incident and damages, 
calculating the estimated and actual loss amounts, checking the policy coverage and deductibles, determining the claim status, and finalizing the claim settlement. 
The goal is to provide a clear and comprehensive sequence of actions necessary to process this claim effectively. Along with it generate a refernce based on customer name and customer ID
```{prompt_name}```
"""
    return prompt

def policy_checking(prompt_name):
    prompt = f"""
Using the detailed insurance claim description provided below, perform a policy check to ensure that the incident type is covered under the policy. 
Evaluate whether the policyholder meets all the necessary conditions, such as coverage limits, deductibles, and policy validity. 
The result should confirm if the policy supports the claim, and identify any potential issues or gaps in coverage that could affect the claim's outcome. Along with it generate a refernce based on customer name and customer ID
```{prompt_name}```
"""
    return prompt

