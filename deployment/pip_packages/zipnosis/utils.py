import re
import logging
import pkg_resources
from zipnosis.setup_logger import logger

def na_fill_string(func, fill_val=''):
    '''
    wrapper that returns (None, None) if the even that a method fails

    '''
    def handler(*args, **kwargs):
        try:
            output =func(*args, **kwargs)
            msg = 'method {} suceeed'.format(func)
            logger.debug(msg)
            return output
        except Exception as e:
            msg = 'method {0} failed with {1}'.format(func, e)
            logger.warning(msg)
            return fill_val
    return handler

def na_fill_none(func, fill_val=None):
    '''
    wrapper that returns (None, None) if the even that a method fails

    '''
    def handler(*args, **kwargs):
        try:
            output =func(*args, **kwargs)
            msg = 'method {} suceeed'.format(func)
            logger.debug(msg)
            return output
        except Exception as e:
            msg = 'method {0} failed with {1}'.format(func, e)
            logger.warning(msg)
            return fill_val
    return handler


@na_fill_string
def anonymize(text):
    # A method to remove all occurrances of a patients' name.
    # Replace address with token
    address = re.search(r'Patient Address:(.*?) \s', text).group(0)[17:]
    text = text.replace(address, "<Address Replaced> ")

    # Replace phone number with token
    phone_number = re.search(r'Patient Phone:(.*?) \s', text).group(0)[15:]
    text = text.replace(phone_number, "<Phone Replaced> ")

    # Replace DOB with token
    dob = re.search(r'Patient DOB:(.*?) \s', text).group(0)[13:]
    text = text.replace(dob, "<DOB Replaced> ")

    # Replace patient name with token
    name = re.search(r'Patient:(.*?) \s', text).group(0)[8:].split()
    # Need to check length of name because some have middle name
    # ie 'John Smith' vs 'John C Smith'
    if len(name) == 2:
        text = text.replace(name[0], "<First Name>").replace(name[1], "<Last Name>")
    if len(name) == 3:
        # If middle/second name appears, it also appears in rest of document
        # ie 'Mary Lou' or 'Mary L' appears throughout
        first_middle = name[0] + ' ' + name[1]
        text = text.replace(first_middle, "<First Name>").replace(name[2], "<Last Initial>")
    text = text.lower()
    return text


@na_fill_string
def get_patient_free_text(text):
    return re.findall('(?<= \\(free text\\):).*?(?=\s\s\s\s)', text.lower())[0]


@na_fill_string
def get_travel_free_text(text):
    text = re.sub('  ', ' ', text)
    text = re.sub('   ', ' ', text)
    pattern = '(?<=countries or locations traveled as reported by the patient \\(free text\\):).*?(?=<first name>)'
    return re.findall(pattern, text.lower())[0]


@na_fill_string
def get_patient_information(text):
    text = re.sub('\s+', ' ', text)
    pattern = '(?<=patient summary)(.*)(?=when asked the question)'
    return re.findall(pattern, text.lower())[0]

@na_fill_string
def get_pertinent_covid19_information(text):
    pattern = '(?<=pertinent covid-19 \(coronavirus\) information)(.*)(?=pertinent medical history)'
    return re.findall(  pattern, text.lower())[0]


@na_fill_string
def get_symptoms(text):

        # Two versions of symptom details - this accounts for both
        v1 = re.findall('(?<=symptom details)(.*)(?=precipitating events)', text.lower())
        v2 = re.findall('(?<=symptom details)(.*)(?=pertinent covid)', text.lower())
        if v1:
            return v1[0]
        elif v2:
            return v2[0]
@na_fill_string
def get_diagnosis(x):
    pattern = '(?<=diagnosis:).+(?=icd)'
    x = re.findall(pattern , x)[0]
    if 'icd' in x:
        return re.split('icd', x)[0]
    else:
        return x

@na_fill_none
def extractSex(patientSummaryText):
    if 'female' in patientSummaryText:
        return 1
    else:
        return 0

@na_fill_none
def extractAge(patientSummaryText):
    age_sentence = re.findall('(?<=is a)(.*)(?=year old)', patientSummaryText.lower())[0]
    return float(re.findall(r'\d+', age_sentence)[0])


@na_fill_none
def extractWeight(text):
    weight_text = re.search(r'Weight:(.*?) \s', text).group(0)
    return int(re.findall(r'\d+', weight_text)[0])


@na_fill_none
def extractTravel(tup):
    freePatientText, pertinentCovid19Text = tup
    if 'has traveled' in pertinentCovid19Text:
        return 1
    if 'travel' in freePatientText or 'traveled' in freePatientText:
        return 1
    else:
        return 0

@na_fill_none
def extractTemperature(text):
    # This method only accounts for temperature recorded in the symptom section OR a denial of the presence of fever by patient
    try:
        temp_sentence = re.findall(r'\\* temperature: .+ fahrenheit\.', text)[0].split('. ')[0]
        floats = re.findall(r'\d+\.\d+', temp_sentence)
        ints = re.findall(r'\d+', temp_sentence)
        if floats:
            return floats[0]
        else:
            return ints[0]
    except:
        return get_denied_fever(text)

@na_fill_none
def get_denied_fever(text):

    # Get sentence where patient denies symptoms and return 1 if fever in list, otherwise return 0
    return 98.6 if 'fever' in re.findall(r'\<first name> denies .+ .', text)[0].split('. ')[0] else None

@na_fill_string
def get_symptom(section_symptoms, x):
    x = x.replace('_', ' ')  # remove underbar in column name bc that's how it appears in text
    symptoms_split = section_symptoms.split('. ')
    for symp in symptoms_split:
        try:
            # try to find a "* symptom" header - break out of for loop otherwise
            # will fail if you don't use try / except
            current_symp = re.findall(r'\* (.*):', symp)[0]
        except:
            break
        if current_symp == x:
            return 1
    return 0


@na_fill_string
def extractICD(x):
    x = re.findall('(?<=icd:) ?[a-z][0-9]+[.]?[0-9]+', x)[0]
    if 'icd' in x:
        return re.split('icd', x)[0].strip()
    else:
        return x


@na_fill_none
def extractICDPrefix(x):
    x = re.findall('(?<=icd:) ?[a-z]', x)[0]
    if 'icd' in x:
        return re.split('icd', x)[0].strip()
    else:
        return x


@na_fill_string
def get_diagnosis(text):
    return re.findall('(?<=diagnosis:)(.*)(?=icd:)', text.lower())[0]

@na_fill_none
def extractSymptomOnsetOrdinal(text):
    groups_order = ['today', '1-2 days ago', '3-6 days ago', '7-9 days ago', '10-13 days ago', '2-3 weeks ago',
                    '1 month or more ago']
    pattern = 'states (his|her) symptoms started(\\sgradually\\s|\\ssuddenly\\s|\\s)(today|1-2 days ago|3-6 days ago|7-9 days ago|10-13 days ago|2-3 weeks ago|1 month or more ago)'
    symptom_sentence = re.search(pattern, text).group(0)
    days = re.search(
        '(today|1-2 days ago|3-6 days ago|7-9 days ago|10-13 days ago|2-3 weeks ago|1 month or more ago)',
        symptom_sentence).group(0)
    quickness = re.search('(gradually|suddenly)', symptom_sentence)
    if quickness:
        return 1 if quickness.group(0) == "suddenly" else 0
    elif days == "today":
        return 1
    else:
        return 0

@na_fill_none
def extractSymptomStart(text):
    # returns index of when the symptom started (Ex. - 0 -> today, 1 -> 1-2 days ago, etc...)
    try:
        groups_order = ['today', '1-2 days ago', '3-6 days ago', '7-9 days ago', '10-13 days ago', '2-3 weeks ago',
                        '1 month or more ago']
        pattern = 'states (his|her) symptoms started(\\sgradually\\s|\\ssuddenly\\s|\\s)(today|1-2 days ago|3-6 days ago|7-9 days ago|10-13 days ago|2-3 weeks ago|1 month or more ago)'
        symptom_sentence = re.search(pattern, text).group(0)
        days = re.search(
            '(today|1-2 days ago|3-6 days ago|7-9 days ago|10-13 days ago|2-3 weeks ago|1 month or more ago)',
            symptom_sentence).group(0)
        quickness = re.search('(gradually|suddenly)', symptom_sentence)
        if quickness:
            return groups_order.index(days)
        elif days == "today":
            return groups_order.index(days)
        else:
            return groups_order.index(days)
    except:
        return 7

@na_fill_none
def extractHealthcareWorker(text):
    text = re.sub('\s+', ' ', text)
    pattern = '((is|is not|is either) a healthcare worker|does not work or volunteer as healthcare worker|either works or volunteers as a healthcare worker)'
    found = re.search(pattern, text).group(0)
    if 'is not' in found or 'does not' in found:
        return 0
    elif 'is a' in found or 'either works':
        return 1


@na_fill_none
def extractShortnessOfBreath(text):
    # is experiencing difficulty breathing due to nasal congestion but she is not short of breath --> 0
    # is not experiencing dyspnea --> 0
    # is experiencing difficulty breathing due to nasal congestion but she is not short of breath --> 0
    # is experiencing mild difficulty breathing with activities but can speak normally in full sentences --> 1
    # difficulty breathing even when resting --> 1

    text = re.sub('\s+', ' ', text)
    pattern = '((is|is not) experiencing(\\smild|) (difficulty breathing|dyspnea)(.*|.)|(\* difficulty breathing even when resting|not experiencing dyspnea))'
    db = re.search(pattern, text).group(0).split('. ')[0]
    if 'experiencing mild difficulty breathing with activities but can speak normally in full sentences' in db or '* difficulty breathing' in db:
        return 1
    else:
        return 0

@na_fill_none
def extractDifficultyBreathingText(text):
    text = re.sub('\s+', ' ', text)
    pattern = '((is|is not) experiencing(\\smild|) (difficulty breathing|dyspnea)(.*|.)|(\* difficulty breathing even when resting|not experiencing dyspnea))'
    db = re.search(pattern, text).group(0).split('. ')[0]
    return db

@na_fill_none
def extractDeniedSymptoms(text):
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^.A-Za-z0-9<> ]+', '', text)
    pattern = '<first name> denies having (.*). (he|she) also denies (.*).'
    capture = re.search(pattern, text).group(0).split('. ')
    return capture[0] + '. ' + capture[1]


@na_fill_none
def extractSmokes(text):
    pattern = '<first name> (smokes or uses|does not smoke or use) smokeless tobacco.'
    return 1 if 'smokes or uses' in re.search(pattern, text).group(0) else 0


## Missing 28
@na_fill_none
def extractZip(text):
    address = re.search(r'Patient Address:(.*?) \s', text).group(0)[17:]
    zip_plus4 = address.split()[-1].split('-')[0]  # int(re.findall(r'\d+', ' "29588"')[0])
    return zip_plus4 if len(zip_plus4) == 5 else None


@na_fill_none
def extractLabConfirmedContact(text):
    text = re.sub('\s+', ' ', text)
    pattern = '(has|has not) had (a\\s|)close contact with a laboratory(-|\\s)confirmed (positive\\s|)covid-19 patient within 14 days of symptom onset.'
    string = re.search(pattern, text).group(0)
    if 'has had a close contact' in string or 'has had close contact':
        return 1
    else:
        return 0


@na_fill_none
def extractSuspectedContact(text):
    text = re.sub('\s+', ' ', text)
    pattern = '(has|has not) had a close contact with a suspected covid-19 patient within 14 days of symptom onset'
    string = re.search(pattern, text).group(0)
    if 'has had a close contact' in string:
        return 1
    else:
        return 0


@na_fill_none
def extractPregnancy(text):
    text = re.sub('\s+', ' ', text)
    pattern = '(is pregnant|is not sure if she is pregnant|denies pregnancy)'  # she is not sure if she is pregnant and denies breastfeeding
    string = re.search(pattern, text).group(0)
    if 'denies pregnancy' in string:
        return 0
    else:
        return 1

def get_test_note():
    path = 'resources//test_note'
    filepath = pkg_resources.resource_filename('zipnosis', path)
    with open(filepath, 'r') as f:
        text = f.readlines()
    return text[0]


def _test_utils():
    x = get_test_note()
    x_anomous = anonymize(x)
    x_symptoms = get_symptoms(x)
    assert ' <first name> ' in x_anomous
    assert 'i am  currently taking' in get_patient_free_text(x)
    get_travel_free_text(x)
    assert 'jane has not had close contact with a laboratory' in get_pertinent_covid19_information(x)
    assert 'cough: jane coughs' in get_symptoms(x)
    assert 0 == extractTravel((get_patient_free_text(x), get_pertinent_covid19_information(x) ))
    assert 1== extractSex(get_patient_information(x))
    assert 42.0 ==  extractAge(get_patient_information(x))
    assert 128 == extractWeight(x)
    assert 98.6 == extractTemperature(x_anomous)
    assert ' j02.9' == extractICD(x_anomous)
    assert ' j' == extractICDPrefix(x_anomous)
    extractHealthcareWorker(x_anomous)
    assert 1 ==  extractShortnessOfBreath(x_anomous)
    assert 0 ==  extractSmokes(x_anomous)
    assert 1 == extractLabConfirmedContact(x_anomous)
    extractSuspectedContact(x_anomous)
    assert 0 == extractPregnancy(x_anomous)
    assert '29407' == extractZip(x)
    assert 0 == extractSymptomOnsetOrdinal(x_anomous)
    assert 2 == extractSymptomStart(x_anomous)
    assert 1 == get_symptom(x_symptoms, 'cough')
    assert 0 == get_symptom(x_symptoms, 'facial_pain_or_pressure')
    assert 0 == get_symptom(x_symptoms, 'headache')
    assert 0 == get_symptom(x_symptoms, 'nasal_secretions')
    assert 0 == get_symptom(x_symptoms, 'temperature_symptom')
    assert 0 == get_symptom(x_symptoms, 'sore_throat')
    logger.info('zipnosis test_utils complete')
