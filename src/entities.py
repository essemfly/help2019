class PersonalRecord:
    def __init__(self, datetime, birth_date):
        self.measurement_datetime = datetime
        self.time_from_birth = birth_date
        self.weight = None
        self.height = None
        self.temp = None
        self.IDBP = None
        self.IMBP = None
        self.ISBP = None
        self.FDBP = None
        self.FMBP = None
        self.FSBP = None
        self.BT = None
        self.CVP = None
        self.ETCO2 = None
        self.PR = None
        self.LAP = None
        self.MINUTE_VOLUME = None
        self.PMEAN = None
        self.DBP = None
        self.MBP = None
        self.SBP = None
        self.DPAP = None
        self.MPAP = None
        self.SPAP = None
        self.PPEAK = None
        self.RR = None
        self.FREQ_MEASURE = None
        self.SPO2 = None
        self.VTE = None
        self.VIT = None

    def set(self, concept_id, source, value):
        if concept_id == 4099154.0:
            self.weight = value
        elif concept_id == 4177340.0:
            self.height = value
        elif concept_id == 4302666.0:
            self.temp = value
        elif concept_id == 4239408.0:
            self.PR = value
        elif concept_id == 4313591.0:
            self.RR = value
        elif concept_id == 4011919.0:
            self.SPO2 = value
        elif concept_id == 4068414.0:
            self.DBP = value
        elif concept_id == 4239021.0:
            self.MBP = value
        elif concept_id == 4354252.0:
            self.SBP = value
        elif concept_id == 4354253.0:
            self.IDBP = value
        elif concept_id == 4108290.0:
            self.IMBP = value
        elif concept_id == 4353843.0:
            self.ISBP = value
        else:
            print('Missed concept ID: ', f'{concept_id}-{source}')
            pass
