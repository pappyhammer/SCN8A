import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from scipy import stats
from sortedcontainers import SortedDict
from datetime import datetime
from matplotlib import rcParams
from bisect import bisect_right
import math
import random
from matplotlib.patches import Polygon

import os


class EpilepsiaColor:
    colors = {"rose pale": "#e4b8b4",
              "rose normal": "#ce8080",
              "rose fonce": "#a30234",
              "rose fonce++": "#511c23",
              "orange clair": "#f1b682",
              "orange fonce": "#e37c1d",
              "jaune": "#ffde75",
              "vert clair": "#abb47d",
              "vert fonce": "#677719",
              "vert emeraude clair": "#a1c5cb",
              "vert emeraude normal": "#5698a3",
              "vert emeraude fonce": "#00545e",
              "vert emeraude fonce++": "#002e30",
              "bleu clair": "#bacfec",
              "bleu normal": "#0076c0",
              "bleu fonce": "#002157",
              "bleu mauve": "#7a5071"}

    def __init__(self):
        pass


class SCN8APatient:
    mut_function_code = {-1: "NA", 0: "missense", 1: "splice-site", 2: "frameshift", 3: "nonsense"}

    def __init__(self, series):
        """
        series from cvs data_frame, with aa.change.position field
        :param data_frame:
        """
        self.pandas_series = series
        self.aa_change_position = series["aa.change.position"]
        self.msc = True if series["msc"] == 1 else False
        self.cohort = series["cohort"]
        self.patient_id = series["patient"]
        self.age_onset = int(series["age.onset"])
        self.ofc_birth = series["ofc.birth"]
        self.ofc_evolution = series["ofc.evolution"]
        self.mri_onset = series["mri.onset"]
        self.mri_followup = series["mri.followup"]
        self.eeg_onset = series["EEG.onset"]
        self.delay = series["delay"]
        self.group_1 = True if series["smooth.mode"] == 1 else False
        self.group_1_NA = True if series["smooth.mode"] == -1 else False
        self.group_2 = True if series["sharp.mode"] == 1 else False
        self.group_2_NA = True if series["sharp.mode"] == -1 else False
        self.group_1_a = True if series["un.a"] == 1 else False
        self.group_1_b = True if series["un.b"] == 1 else False
        self.seizure_type_onset = series["seizure.type.onset"]
        self.dev_after = series["dev.after"]
        self.dev_before = series["dev.before"]
        self.sudep = series["sudep"]
        self.aed_efficacy = series["AED.efficacy"]

        self.scb_names = ["PHT", "CBZ", "OXC", "LCM", "LTG", "ZON"]
        self.scb_names_reduced = ["PHT", "CBZ", "OXC", "LTG"]
        # self.scb_names = self.scb_names_reduced
        self.non_scb_aed_names = ["TPM", "VPA", "LEV", "CLN", "ACTH", "LD", "MDL", "GBP", "KD", "PB", "LCM", "VGB",
                                  "nitrazepam",
                                  "CLB", "RUF", "STM", "PGN", "HC", "CANNABIDIOL"]
        self.aed_names = self.scb_names + self.non_scb_aed_names
        self.aed_efficacy_dict = dict()
        for aed in self.aed_names:
            self.aed_efficacy_dict[aed] = series[aed]

        self.dna_change = series["dna.change"]
        self.aa_change = str(series["aa.change"])
        self.aa_change = self.aa_change.replace(" ", "")

        self.dna_change = series["dna.change"]

        aa_one_letter_tuples = [("ala", "A"), ("arg", "R"), ("asn", "R"), ("asp", "D"),
                                ("asx", "B"), ("cys", "C"), ("glu", "E"), ("gln", "Q"),
                                ("glx", "Z"), ("gly", "G"), ("his", "H"), ("ile", "I"),
                                ("leu", "L"), ("lys", "K"), ("met", "M"), ("phe", "F"),
                                ("pro", "P"), ("ser", "S"), ("thr", "T"), ("trp", "W"),
                                ("tyr", "Y"), ("val", "V")]
        self.aa_to_one_letter_dict = dict()
        self.one_letter_to_aa_dict = dict()
        for aa_tuple in aa_one_letter_tuples:
            self.aa_to_one_letter_dict[aa_tuple[0]] = aa_tuple[1]
            self.one_letter_to_aa_dict[aa_tuple[1]] = aa_tuple[0]

        self.mut_function = series["mut.function"]

        if self.aa_change != "nan":
            # special case for non-sens mutation
            if self.mut_function == 3:
                # format is : Arg1820*
                first_letter = self.aa_change[:3].lower()
                pos = self.aa_change[3:-1]
                self.aa_change_one_letter = self.aa_to_one_letter_dict[first_letter] + pos + "*"
                # print(f"self.aa_change_one_letter {self.aa_change_one_letter}")
            # special case if non missense mutation
            elif self.aa_change[-3:] == "del":
                first_letter = self.aa_change[:3].lower()
                second_letter = self.aa_change[8:11].lower()
                pos1 = self.aa_change[3:7]
                pos2 = self.aa_change[11:15]
                self.aa_change_one_letter = self.aa_to_one_letter_dict[first_letter] + pos1 + "_" + \
                                            self.aa_to_one_letter_dict[second_letter] + pos2 + "del"
            elif self.aa_change[:1] == "c":
                # it means aa change is unknow, and the dna change is displayed instead
                self.aa_change_one_letter = self.aa_change
            else:
                first_letter = self.aa_change[:3].lower()
                second_letter = self.aa_change[-3:].lower()
                pos = self.aa_change[3:-3]
                self.aa_change_one_letter = self.aa_to_one_letter_dict[first_letter] + pos + \
                                            self.aa_to_one_letter_dict[second_letter]
        else:
            self.aa_change_one_letter = None

        # use to filter mutations to display according to some criteria, if this information is not available for
        # this patient, then the attribute get to True
        self.NA_filtered_data = False
        # region will be updated when calling nav16 build_region function
        self.region = None

    def get_a_copy(self):
        return SCN8APatient(self.pandas_series)

    def __str__(self):

        if self.patient_id is np.nan:
            id_p = ""
        else:
            id_p = " " + str(self.patient_id)
        return str(self.cohort) + id_p

    def __hash__(self):
        # str() in cas the variable value is NaN
        return hash(str(self.cohort) + str(self.patient_id))

    def __eq__(self, other):
        return (str(self.cohort) + str(self.patient_id)) == (str(other.cohort) + str(other.patient_id))


class CanalRegion:
    """
    Represents one or more CanalElement
    Used to apply stats between each element
    """

    def __init__(self, region_id, canal_elements, color):
        self.region_id = region_id
        # contains instances of SCN8APatients
        self.patients = set()
        # key is an int representing the aa position, value is a list of patients
        self.mutations_by_aa = SortedDict()
        # total number of aa
        self.nb_total_aa = 0
        # total number of aa with a mutation
        self.nb_aa_with_mutation = 0
        self.color = color
        self.canal_elements = canal_elements

        for canal_element in canal_elements:
            patients_mut = canal_element.get_mutations()
            for patient in patients_mut:
                self.patients.add(patient)
                patient.region = self

            for aa_pos, patients_mut in canal_element.mutations.items():
                if aa_pos not in self.mutations_by_aa:
                    self.mutations_by_aa[aa_pos] = set()
                for patient in patients_mut:
                    self.mutations_by_aa[aa_pos].add(patient)

            self.nb_total_aa += canal_element.nb_aa
            canal_element.region = self

        self.nb_aa_with_mutation = len(self.mutations_by_aa)
        self.nb_mutated_patients = 0
        for patients in self.mutations_by_aa.values():
            self.nb_mutated_patients += len(patients)
        self.variants_frequency = self.nb_aa_with_mutation / self.nb_total_aa
        # if self.region_id == "N":
        #     print(f"N: self.nb_aa_with_mutation {self.nb_aa_with_mutation}, "
        #           f"self.variants_frequency {self.variants_frequency}")

    # consider a region identical to another if they have the same ID
    def __hash__(self):
        # str() in cas the variable value is NaN
        return hash(self.region_id)

    def __eq__(self, other):
        return self.region_id == self.region_id


class CanalElement:
    def __init__(self, first_position, last_position, start_coord=None, end_coord=None, previous_element=None,
                 next_element=None,
                 no_elements_inside=True, parent=None):
        self.first_position = first_position
        self.last_position = last_position
        self.nb_aa = self.last_position - self.first_position

        self.no_elements_inside = no_elements_inside

        # for each position contains in the element, return the element corresponding could be itself or another
        #  (segment, loop etc...)
        self.elements_position = dict()
        if no_elements_inside:
            for i in np.arange(self.first_position, self.last_position + 1):
                self.elements_position[i] = [self]

        self.start_coord = start_coord
        self.end_coord = end_coord

        self.previous_element = None
        self.next_element = None
        if previous_element is not None:
            self.set_previous_element(canal_element=previous_element)
        if next_element is not None:
            self.set_next_element(canal_element=next_element)

        self.parent = parent
        if parent is not None:
            parent.fill_elements_position(canal_element=self)

        # key is the aa position of the mutation, value is a SCN8APatient instance list
        self.mutations = SortedDict()

        # ------------ plotting attributes -----------
        self.total_width = 1020  # self.end_coord[0]
        self.total_height = 100  # self.end_coord[1]
        self.plotted = False
        self.segment_line_color = "black"
        self.mutation_scatter_line_width = 1.5
        self.mutation_scatter_size = 70
        self.missense_mutation_marker = "o"
        # self.non_recurrent_mutation_marker = "o"
        # square
        self.frameshift_mutation_marker = "*"
        self.splice_mutation_marker = "s"
        self.NA_mutation_marker = "^"
        # losange
        self.nonsens_mutation_marker = "d"
        self.short_loop_line_color = "black"
        self.short_loop_line_width = 1.5
        self.scatter_alpha = 0.9
        self.scatter_NA_alpha = 0.5
        self.legend_mut_line_color = "black"  # EpilepsiaColor.colors["rose normal"]
        self.legend_mut_line_width = 0.8
        self.legend_mut_z_order = 12
        self.legend_mut_linestyle = 'dashed'
        # means we display one aa_change for each mutation (except when filtering), if False,
        # we can display something different
        # for each patient instead
        self.display_only_aa_change = True
        self.scatter_msc_color = EpilepsiaColor.colors["bleu normal"]
        self.na_filtered_data_color = "lightgrey"
        self.scatter_color = EpilepsiaColor.colors["vert emeraude fonce"]
        self.legend_mut_color = self.scatter_color  # "black"

        # coordinates where a mutation can be plotted, used to draw line elements as well
        # self.x_mut_coords = None
        # self.y_mut_coords = None
        # how many point to interpolate where the mutation will be located
        # correpond to len(x_mut_coords) and len(y_mut_coords)
        self.len_interpol_mut = 100

        self.region = None

        # ------------ end plotting attributes -----------

    def get_variants_freq(self):
        return self.nb_unique_aa_mutated() / self.nb_aa

    def fill_elements_position(self, canal_element):
        start_pos = max(canal_element.first_position, self.first_position)
        end_pos = min(canal_element.last_position, self.last_position)
        for i in np.arange(start_pos, end_pos + 1):
            if i in self.elements_position:
                self.elements_position[i].append(canal_element)
                # print(f"{self} position {i} {str(self.elements_position[i])}")
            # if i == 1:
            #     print(f"i==1 {canal_element}, self {self}")
            self.elements_position[i] = [canal_element]

    def set_previous_element(self, canal_element):
        self.previous_element = canal_element
        if not self.no_elements_inside:
            self.fill_elements_position(canal_element=canal_element)

    def get_older_parent(self):
        if self.parent is not None:
            return self.parent.get_older_parent()
        return self

    def set_next_element(self, canal_element):
        self.next_element = canal_element
        if not self.no_elements_inside:
            self.fill_elements_position(canal_element=canal_element)

    def get_element_at_position(self, position, top_element=False):
        """

        :param position: aa
        :param top_element: means you return the element containing the other ones
        :return:
        """
        element = self.elements_position[position][0]
        if element.no_elements_inside or top_element:
            return element
        else:
            return element.get_element_at_position(position)

    def get_first_element(self, different_from=None):
        if different_from is not None:
            for i in np.arange(self.first_position, self.last_position + 1):
                element = self.get_element_at_position(position=i)
                if str(element) != str(different_from):
                    return element
            return None
        else:
            return self.get_element_at_position(position=self.first_position)

    def __str__(self):
        return "CanalElement"

    def __eq__(self, other):
        return str(self) == str(other)

    def get_str_structure(self):
        result = ""
        # print(f"get {str(self)}, self.no_elements_inside {self.no_elements_inside}, "
        #       f"self.next_element {self.next_element}")
        if self.no_elements_inside:
            if self.next_element is not None:
                jump_line = " - "
                if self.parent != self.next_element.parent:
                    jump_line = "\n"
                # print(f"next: {self.next_element.get_str_structure()}")
                result = str(self) + jump_line + self.next_element.get_str_structure()
            else:
                result = str(self)
        else:
            result += "\n"
            result += self.get_first_element(different_from=self.previous_element).get_str_structure()
        return result

    def add_mutations(self, patients_mut):
        """

        :param patients_mut: dict with key aa position, and value a list of one or more SCN8APatient
        :return:
        """
        for aa_pos, v in patients_mut.items():
            # v is a list of patients
            if (aa_pos >= self.first_position) and (aa_pos <= self.last_position):
                # test if there is already a patient mutated known at this position
                if aa_pos in self.mutations:
                    self.mutations[aa_pos].extend(v)
                else:
                    self.mutations[aa_pos] = v
                # add the mutation to elements contains in this element.
                for element in self.elements_position[aa_pos]:
                    if element != self:
                        tmp_dict = {aa_pos: v}
                        element.add_mutations(tmp_dict)

    # ---------- ploting function -----------
    @staticmethod
    def interpolate_fct(x, y):
        return interp1d(x, y, kind='cubic', )

    def get_scatter_colors(self, patients):
        """
        :param others:
        list of others patients, empty list is the only patient with this mutation
        :return:
        """
        fct_mut_dict = dict()
        for mut in SCN8APatient.mut_function_code.values():
            fct_mut_dict[mut] = {"nb_patients": 0, "all_NA": True, "from_msc": False}

        for patient in patients:
            # type of mutation: missense, frameshift, etc...
            mut_fct = patient.mut_function
            if pd.isnull(mut_fct):
                mut_fct = -1
            key_mut = SCN8APatient.mut_function_code[mut_fct]
            fct_mut_dict[key_mut]["nb_patients"] += 1

            if not patient.NA_filtered_data:
                fct_mut_dict[key_mut]["all_NA"] = False
            if patient.msc:
                fct_mut_dict[key_mut]["from_msc"] = True

        list_param = []

        for k_mut, v in fct_mut_dict.items():
            if v["nb_patients"] == 0:
                continue
            face_color = self.scatter_color
            edge_color = self.scatter_color
            z_order = 9

            if v["from_msc"]:
                face_color = self.scatter_msc_color
                edge_color = self.scatter_msc_color
                z_order = 10

            if v["all_NA"]:
                face_color = self.na_filtered_data_color
                edge_color = self.na_filtered_data_color
                alpha_value = self.scatter_NA_alpha
            else:
                if len(fct_mut_dict) > 1:
                    alpha_value = self.scatter_alpha
                else:
                    alpha_value = self.scatter_alpha

            if v["nb_patients"] > 1:
                face_color = "none"

            if k_mut == "NA":
                marker = self.NA_mutation_marker
            elif k_mut == "frameshift":
                marker = self.frameshift_mutation_marker
            elif k_mut == "nonsense":
                marker = self.nonsens_mutation_marker
            elif k_mut == "missense":
                marker = self.missense_mutation_marker
            elif k_mut == "splice-site":
                marker = self.splice_mutation_marker

            list_param.append((marker, face_color, edge_color, z_order, alpha_value))

        return list_param

    def plot_next_element(self):
        if self.no_elements_inside:
            if self.next_element is not None:
                self.next_element.plot_element()
        else:
            self.get_first_element(different_from=self.previous_element).plot_element()

    def plot_element(self, plot_next=True):
        if plot_next:
            self.plot_next_element()

    def get_coord_from_aa_position(self, aa_position):
        # temporary action, for debugging
        if self.x_mut_coords is None:
            print("self.x_mut_coords  is None")
            return None

        # should return a tuple of float, coresponding to x and y coordinates
        if aa_position < self.first_position or aa_position > self.last_position:
            return None

        # distance percentage between first and last position
        perc = aa_position - self.first_position
        perc /= (self.last_position - self.first_position)

        index_coord = int((len(self.x_mut_coords) - 1) * perc)
        x_coord = self.x_mut_coords[index_coord]

        y_coord = self.y_mut_coords[index_coord]

        return tuple([x_coord, y_coord])

    # ---------- end ploting function -----------

    # ------- for debugging ------------
    def is_it_full(self):
        for i in np.arange(self.first_position, self.last_position + 1):
            if i not in self.elements_position:
                return False
        return True

    def str_representation_elements_at_position(self, aa_position):
        result = ""
        if aa_position not in self.elements_position:
            return result
        for element in self.elements_position[aa_position]:
            if element == self:
                result += f"{str(self)}"
            else:
                result += f"[{str(self)} - {element.str_representation_elements_at_position(aa_position)}]"
        return result

    def print_mutations(self, sorted=True):
        if sorted:
            for aa_pos in np.arange(self.first_position, self.last_position + 1):
                if aa_pos not in self.mutations:
                    continue
                patients = self.mutations[aa_pos]
                nb_patients = len(patients)
                pa_str = "patient"
                if nb_patients > 1:
                    pa_str += "s"
                patient_str = ""
                nb_patient_msc = 0
                for patient in patients:
                    if patient.msc:
                        nb_patient_msc += 1
                    patient_str += str(patient) + " "
                patient_str = patient_str[:-1]
                msc_str = ""
                if nb_patient_msc > 0:
                    msc_str = f"({nb_patient_msc})"
                # print(f"AA pos: {aa_pos}, {self.str_representation_elements_at_position(aa_pos)}, "
                #       f"{nb_patients} {pa_str}{msc_str}: {patient_str}")
        else:
            for aa_pos, patients in self.mutations.items():
                nb_patients = len(patients)
                pa_str = "patient"
                if nb_patients > 1:
                    pa_str += "s"

                patient_str = ""
                nb_patient_msc = 0
                for patient in patients:
                    if patient.msc:
                        nb_patient_msc += 1
                    patient_str += str(patient) + " "
                patient_str = patient_str[:-1]
                msc_str = ""
                if nb_patient_msc > 0:
                    msc_str = f" ({msc_str})"
                print(f"AA pos: {aa_pos}, {self.str_representation_elements_at_position(aa_pos)}, "
                      f"{nb_patients} {pa_str}{msc_str}: {patient_str}")

    # ------- end for debugging ------------

    def get_mutations(self):
        result = []
        for aa_pos in np.arange(self.first_position, self.last_position + 1):
            if aa_pos not in self.mutations:
                continue
            patients = self.mutations[aa_pos]
            result.extend(patients)
        return result

    def nb_unique_aa_mutated(self):
        return len(self.mutations)

    def patients_grouped_by_unique_mutation(self):
        # different of nb_unique_aa_mutated, as for a same position, there could be different aa change
        nb_mut = 0
        # list_patients need ot be split in one or more list, in order for patients with
        # the same aa_pos but different aa change to be displayed on a different slot
        original_list_patients = []
        for aa_pos, list_patients in self.mutations.items():
            nb_patients_total = len(list_patients)

            for patient in list_patients:
                patient_added = False
                if len(original_list_patients) == 0:
                    original_list_patients.append([patient])
                else:
                    for sub_list in original_list_patients:
                        patient_sub = sub_list[0]
                        if patient_sub.aa_change.lower() == patient.aa_change.lower():
                            sub_list.append(patient)
                            patient_added = True
                    if not patient_added:
                        original_list_patients.append([patient])
            nb_mut += len(original_list_patients)

        return original_list_patients

    def nb_mutations(self):
        return len(self.get_mutations())


class Nav16(CanalElement):
    def __init__(self, fig):
        super().__init__(start_coord=(30, 8), end_coord=(1020, 100), first_position=1, last_position=1980,
                         no_elements_inside=False)
        self.start_coord = (30, 8)
        self.end_coord = (self.total_width, self.total_height)
        # --------------- graphical components ------------
        self.fig = fig
        self.axe_plot = None
        # --------------- end graphical components ------------
        nb_domains = 4
        # four domains
        self.domains = dict()
        len_domain = (self.total_width / 4.2)
        self.domains[1] = Domain(number=1, first_position=114, last_position=442, parent=self,
                                 start_coord=(self.start_coord[0], self.start_coord[1]),
                                 end_coord=(self.start_coord[0] + len_domain - 1, self.start_coord[1]))
        self.domains[2] = Domain(number=2, first_position=735, last_position=1007, parent=self,
                                 start_coord=(self.domains[1].end_coord[0] + 1, self.start_coord[1]),
                                 end_coord=(self.domains[1].end_coord[0] + 1 + len_domain, self.start_coord[1]))
        self.domains[1].set_next_domain(domain=self.domains[2])
        self.domains[3] = Domain(number=3, first_position=1180, last_position=1495, parent=self,
                                 start_coord=(self.domains[2].end_coord[0] + 1, self.start_coord[1]),
                                 end_coord=(self.domains[2].end_coord[0] + len_domain, self.start_coord[1])
                                 )
        self.domains[2].set_next_domain(domain=self.domains[3])
        self.domains[2].set_previous_domain(domain=self.domains[1])
        self.domains[4] = Domain(number=4, first_position=1504, last_position=1801, parent=self,
                                 start_coord=(self.domains[3].end_coord[0] + 1, self.start_coord[1]),
                                 end_coord=(self.domains[3].end_coord[0] + len_domain, self.start_coord[1])
                                 )
        self.domains[3].set_next_domain(domain=self.domains[4])
        self.domains[3].set_previous_domain(domain=self.domains[2])
        self.domains[4].set_previous_domain(domain=self.domains[3])

        # self.pore_loop = PoreLoop(first_position=, last_position=, previous_element=self.domains[3],
        #                                   next_element=self.domains[4]))
        nb_segments_by_domain = 6
        segments_aa_position = dict()
        segments_aa_position[1] = [(133, 151), (159, 179), (194, 211), (218, 234), (254, 273), (388, 408)]
        segments_aa_position[2] = [(754, 772), (784, 803), (818, 837), (840, 857), (874, 892), (956, 976)]
        segments_aa_position[3] = [(1200, 1217), (1231, 1249), (1264, 1282), (1291, 1309), (1327, 1346), (1439, 1460)]
        segments_aa_position[4] = [(1524, 1541), (1553, 1571), (1584, 1601), (1615, 1631), (1651, 1668), (1743, 1765)]
        intra_mb_first_position = dict({1: (356, 380), 2: (922, 942), 3: (1400, 1421), 4: (1691, 1713)})

        for d in np.arange(1, nb_domains + 1):
            segments = []
            short_loop_number = 1
            for s in np.arange(nb_segments_by_domain):
                first_position = segments_aa_position[d][s][0]
                last_position = segments_aa_position[d][s][1]
                segment = Segment(number=s + 1, first_position=first_position, last_position=last_position,
                                  parent=self.domains[d])
                segments.append(segment)
                self.domains[d].add_segment(segment=segment)
                if (s > 0) and (s < 5):
                    ShortLoop(number=short_loop_number, first_position=segments[s - 1].last_position + 1,
                              last_position=first_position - 1,
                              previous_element=segments[s - 1], next_element=segments[s], parent=self.domains[d])
                    short_loop_number += 1
                if s == 5:
                    PoreLoop(first_position=segments[s - 1].last_position + 1, last_position=first_position - 1,
                             intra_mb_first_position=intra_mb_first_position[d][0],
                             intra_mb_last_position=intra_mb_first_position[d][1],
                             previous_element=segments[s - 1], next_element=segments[s], parent=self.domains[d])

        # ----------- Loops between domains  -------------
        self.large_loop_d1_d2 = LargeLoop(number=1, first_position=409, last_position=753,
                                          previous_domain=self.domains[1],
                                          next_domain=self.domains[2], parent=self)

        self.large_loop_d2_d3 = LargeLoop(number=2, first_position=977, last_position=1199,
                                          previous_domain=self.domains[2],
                                          next_domain=self.domains[3], parent=self)

        self.inactivation_gate_loop = InactivationGateLoop(first_position=1461, last_position=1523,
                                                           previous_domain=self.domains[3],
                                                           next_domain=self.domains[4], parent=self)
        # ----------- End loops between domains  -----------

        self.n_terminal = NTerminal(first_position=1, last_position=132, domain=self.domains[1], parent=self)
        self.c_terminal = CTerminal(first_position=1766, last_position=1980, domain=self.domains[4], parent=self)

        # ----------- Start plotting attributes  -----------
        # Used to display aa change for each mutation
        # each key is an int representing a slot, each vlaue is a patient
        self.mutations_display_slots_up = dict()
        self.mutations_display_slots_down = dict()
        # 150 slots
        if self.display_only_aa_change:
            self.nb_slots = 85
        else:
            self.nb_slots = 100
        self.slot_x_coord = np.linspace(5, self.total_width - 5, num=self.nb_slots, endpoint=True)
        self.legend_mut_y_coord_up = (self.total_height * 0.87) + self.start_coord[1]
        self.legend_mut_y_coord_down = (self.total_height * 0.10) + self.start_coord[1]
        self.legend_mut_y_coord_anchorage_up = self.legend_mut_y_coord_up - 0.5
        self.legend_mut_y_coord_anchorage_down = self.legend_mut_y_coord_down + 0.5
        # ----------- End plotting attributes  -----------

        self.regions_dict = dict()
        self.regions_list = []

    def get_slot_for_mut(self, coord_mut, patient, bottom_part):
        """
        Put patient in the good slot and return the slot number
        :param coord_mut:
        :param patient:
        :param bottom_part:
        :return:
        """

        x_coord = coord_mut[0]

        perc = x_coord / self.total_width

        slot = int(self.nb_slots * perc)
        display_slots = self.mutations_display_slots_down if bottom_part else self.mutations_display_slots_up
        if slot not in display_slots:
            display_slots[slot] = patient
        else:
            i = 1
            r = np.random.randint(low=0, high=2)
            up_slot = slot
            down_slot = slot
            # sign = 1 if ((r % 2) == 0) else -1
            # i *= sign
            while True:
                if (up_slot + i) < self.nb_slots:
                    if (up_slot + i) not in display_slots:
                        up_slot = up_slot + i
                        break
                else:
                    up_slot = -1
                    break
                i += 1
            i = 1
            while True:
                if (down_slot - i) > 0:
                    if (down_slot - i) not in display_slots:
                        down_slot = down_slot - i
                        break
                else:
                    down_slot = -1
                    break
                i += 1
            winner_slot = -1
            if up_slot == -1:
                winner_slot = down_slot
            elif down_slot == -1:
                winner_slot = up_slot
            else:
                if np.abs(up_slot - slot) > np.abs(down_slot - slot):
                    winner_slot = down_slot
                else:
                    winner_slot = up_slot
            slot = winner_slot
            display_slots[slot] = patient

        return slot

    def free_slot(self, slot, bottom_part):
        display_slots = self.mutations_display_slots_down if bottom_part else self.mutations_display_slots_up
        if slot in display_slots:
            del display_slots[slot]

    def __str__(self):
        return "NaV1.6"

    def build_regions(self, holland_version=True):
        """
        6 different regions (VSR, pore, TMO, loops, N, and C) for holland_version
        Should be build only after mutations have been added
        :param holland_version: use regions from the Holland paper
        :return: return a dict with key the string representing the region name and as value the RegionCanal instance
        """

        if holland_version:
            n_region = CanalRegion(region_id="N", canal_elements=[self.n_terminal],
                                   color=EpilepsiaColor.colors["rose pale"])

            # Other transmembrane segments and their linking regions were grouped (TMO)
            tmo_elements = []
            for domain in self.domains.values():
                # adding segments 1 to 2 plus the short_loop after each
                for seg in np.arange(1, 3):
                    tmo_elements.append(domain.segments[seg])
                    tmo_elements.append(domain.segments[seg].next_element)
                # adding segment 3
                tmo_elements.append(domain.segments[3])
            tmo_region = CanalRegion(region_id="TMO", canal_elements=tmo_elements,
                                     color=EpilepsiaColor.colors["bleu clair"])

            #  the voltage sensor region (VSR) as S4 and its associated linkers (S3-S4 and S4-S5)
            vsr_elements = []
            for domain in self.domains.values():
                vsr_elements.append(domain.segments[4])
                vsr_elements.append(domain.segments[4].previous_element)
                vsr_elements.append(domain.segments[4].next_element)
            vsr_region = CanalRegion(region_id="VSR", canal_elements=vsr_elements,
                                     color=EpilepsiaColor.colors["jaune"])

            # The pore region was defined as segments S5, S5-S6, S6;
            pore_elements = []
            for domain in self.domains.values():
                pore_elements.append(domain.segments[5])
                # adding the short loop between segments S5 and S6
                pore_elements.append(domain.segments[5].next_element)
                pore_elements.append(domain.segments[6])
            pore_region = CanalRegion(region_id="Pore", canal_elements=pore_elements,
                                      color=EpilepsiaColor.colors["bleu mauve"])

            # intracellular loops linking domains I-IV were grouped together (Loops)
            loops_elements = [self.large_loop_d1_d2, self.large_loop_d2_d3]
            loops_region = CanalRegion(region_id="Loops", canal_elements=loops_elements,
                                       color=EpilepsiaColor.colors["vert clair"])

            inactivation_gate_region = CanalRegion(region_id="IG", canal_elements=[self.inactivation_gate_loop],
                                       color=EpilepsiaColor.colors["orange fonce"])

            c_region = CanalRegion(region_id="C", canal_elements=[self.c_terminal],
                                   color=EpilepsiaColor.colors["rose fonce"])

            self.regions_dict[n_region.region_id] = n_region
            self.regions_dict[tmo_region.region_id] = tmo_region
            self.regions_dict[vsr_region.region_id] = vsr_region
            self.regions_dict[pore_region.region_id] = pore_region
            self.regions_dict[loops_region.region_id] = loops_region
            self.regions_dict[inactivation_gate_region.region_id] = inactivation_gate_region
            self.regions_dict[c_region.region_id] = c_region

            self.regions_list = [n_region, tmo_region, vsr_region, pore_region,
                                 loops_region, inactivation_gate_region, c_region]
            return self.regions_list
        else:
            # use Zuberi 2011 (SCN1A) version
            n_region = CanalRegion(region_id="N", canal_elements=[self.n_terminal],
                                   color=EpilepsiaColor.colors["rose pale"])

            # intracellular loops linking domains I-IV were grouped together (Loops)
            loops_elements = [self.large_loop_d1_d2, self.large_loop_d2_d3, self.inactivation_gate_loop]
            loops_region = CanalRegion(region_id="Loops", canal_elements=loops_elements,
                                       color=EpilepsiaColor.colors["rose fonce++"])

            c_region = CanalRegion(region_id="C", canal_elements=[self.c_terminal],
                                   color=EpilepsiaColor.colors["rose fonce"])

            # Other transmembrane segments and their linking regions were grouped (TMO)
            s_elements = dict()
            s_to_s_elements = dict()
            for domain in self.domains.values():
                for n_seg in np.arange(1, 7):
                    if n_seg not in s_elements:
                        s_elements[n_seg] = []
                    s_elements[n_seg].append(domain.segments[n_seg])
                    if n_seg < 6:
                        if n_seg not in s_to_s_elements:
                            s_to_s_elements[n_seg] = []
                        s_to_s_elements[n_seg].append(domain.segments[n_seg].next_element)
                # tmo_elements.append(domain.segments[seg].next_element)
            self.regions_list = [n_region]
            segment_colors = {1: EpilepsiaColor.colors["orange fonce"], 2: EpilepsiaColor.colors["bleu fonce"],
                              3: EpilepsiaColor.colors["vert clair"], 4: EpilepsiaColor.colors["jaune"],
                              5: EpilepsiaColor.colors["bleu clair"], 6: EpilepsiaColor.colors["bleu mauve"]}
            for n_seg in np.arange(1, 7):
                s_region = CanalRegion(region_id=f"S{n_seg}", canal_elements=s_elements[n_seg],
                                       color=segment_colors[n_seg])
                self.regions_dict[s_region.region_id] = s_region
                self.regions_list.append(s_region)
                if n_seg < 6:
                    s_to_s_region = CanalRegion(region_id=f"S{n_seg}-S{n_seg+1}",
                                                canal_elements=s_to_s_elements[n_seg],
                                                color="black")
                    self.regions_dict[s_to_s_region.region_id] = s_to_s_region
                    self.regions_list.append(s_to_s_region)

            self.regions_list.extend([loops_region, c_region])

            self.regions_dict[n_region.region_id] = n_region
            self.regions_dict[loops_region.region_id] = loops_region
            self.regions_dict[c_region.region_id] = c_region
            # The entire protein is divided into 14 subunits: N-terminal, S1 (segment 1), S1–S2 (segment 1–2 linker),
            # S2, S2–S3, S3, S3–S4, S4, S4–S5, S5, S5–S6, S6,
            # linker regions (large intracellular loops linking the 4 homologous domains), and the C-terminal
            return self.regions_list

    def get_regions(self, region_id):
        if region_id not in self.regions_dict:
            return None
        return self.regions_dict[region_id]

    def plot_mutations(self):
        # print("plot mutations")
        for aa_pos, list_patients in self.mutations.items():
            element = self.get_element_at_position(aa_pos)
            coord = element.get_coord_from_aa_position(aa_pos)
            if coord is None:
                continue
            list_param = self.get_scatter_colors(patients=list_patients)
            # checking if there is a patient without epilepsy
            no_epilepsy_patient = False
            for patient in list_patients:
                if patient.seizure_type_onset == -3:
                    no_epilepsy_patient = True
            # print(f'aa_pos {aa_pos}, len(list_patients): {len(list_patients)}, face_color: {face_color}')
            for param in list_param:
                marker, face_color, edge_color, z_order, alpha_value = param
                if no_epilepsy_patient:
                    edge_color = "black"
                plt.scatter(coord[0], coord[1], marker=marker,
                            facecolors=face_color,
                            linewidth=self.mutation_scatter_line_width,
                            edgecolors=edge_color, s=self.mutation_scatter_size, zorder=z_order,
                            alpha=alpha_value)
            # if len(list_patients) > 1:
            #     plt.text(x=coord[0], y=coord[1], s=f"{len(list_patients)}", color="red", zorder=15,
            #              ha='center', va="center", fontsize=6, fontweight='bold')

    def plot_mutation_legends(self):
        # setting latex usage for multicolor / style text
        # rc('text', usetex=True)
        # rc('text.latex', preamble=r'\usepackage{color}')
        # # rc('text.latex.preamble', preamble=r"\usepackage{amsmath}")
        # matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
        for aa_pos, list_patients in self.mutations.items():
            nb_patients_total = len(list_patients)

            element = self.get_element_at_position(aa_pos)
            coord_mut = element.get_coord_from_aa_position(aa_pos)
            if coord_mut is None:
                continue
            # serve to know if we display mutaiton on the down part of up part
            half_threshold = self.legend_mut_y_coord_down + (
                    (self.legend_mut_y_coord_up - self.legend_mut_y_coord_down) / 2)
            # half_threshold = (self.total_height / 2)
            bottom_part = coord_mut[1] < half_threshold

            # list_patients need ot be split in one or more list, in order for patients with
            # the same aa_pos but different aa change to be displayed on a different slot
            original_list_patients = []
            for patient in list_patients:
                patient_added = False
                if len(original_list_patients) == 0:
                    original_list_patients.append([patient])
                else:
                    for sub_list in original_list_patients:
                        patient_sub = sub_list[0]
                        # if aa_pos = 1872:
                        #     print(f"patient_sub_list.aa_change.lower() {patient_sub.aa_change.lower()}, "
                        #           f"patient.aa_change.lower() {patient.aa_change.lower()}")
                        if patient_sub.aa_change.lower() == patient.aa_change.lower():
                            sub_list.append(patient)
                            patient_added = True
                    if not patient_added:
                        original_list_patients.append([patient])
            # print(f"len(original_list_patients) {len(original_list_patients)}")
            for list_patients in original_list_patients:
                nb_patients_na = 0
                nb_patients_known = 0
                msc_in_the_place = False
                msc_in_the_place_NA = False
                msc_str = ""
                msc_str_NA = ""

                for patient in list_patients:
                    if patient.NA_filtered_data:
                        if patient.msc:
                            if not msc_in_the_place:
                                msc_str_NA += f"[{patient.patient_id[8:]}"
                            else:
                                msc_str_NA += f", {patient.patient_id[8:]}"
                            msc_in_the_place_NA = True
                        nb_patients_na += 1
                    else:
                        if patient.msc:
                            if not msc_in_the_place:
                                msc_str += f"[{patient.patient_id[8:]}"
                            else:
                                msc_str += f", {patient.patient_id[8:]}"
                            msc_in_the_place = True
                        nb_patients_known += 1
                if msc_in_the_place:
                    msc_str += "]"
                if msc_in_the_place_NA:
                    msc_str_NA += "]"

                index_count = 0
                index_count_NA = 0
                for patient in list_patients:
                    if (index_count > 0) and (index_count_NA > 0) and self.display_only_aa_change:
                        break
                    if index_count > 0:
                        # only one line by mutation except if the patient has data NA (unknown)
                        if (not patient.NA_filtered_data) and self.display_only_aa_change:
                            continue
                    if (index_count_NA > 0) and patient.NA_filtered_data:
                        continue
                    slot = self.get_slot_for_mut(coord_mut=coord_mut, patient=patient, bottom_part=bottom_part)
                    if slot < 0:
                        print("No more free slot")
                        break
                    # print(f'slot {slot}')
                    x_coord = self.slot_x_coord[slot]
                    y_coord = self.legend_mut_y_coord_anchorage_down if bottom_part \
                        else self.legend_mut_y_coord_anchorage_up

                    # aa_legend = "p."+patient.aa_change
                    aa_legend = patient.aa_change_one_letter
                    multi_p = ""
                    if patient.NA_filtered_data:
                        if nb_patients_na > 1:
                            multi_p = f"(x{nb_patients_na}) "
                    else:
                        if nb_patients_known > 1:
                            multi_p = f"(x{nb_patients_known}) "
                    if patient.NA_filtered_data:
                        if msc_in_the_place_NA:
                            legend = "{m}{aa} {msc}".format(m=multi_p, aa=aa_legend, msc=msc_str_NA)
                        else:
                            legend = "{m}{aa}".format(m=multi_p, aa=aa_legend)
                    else:
                        if msc_in_the_place:
                            legend = "{m}{aa} {msc}".format(m=multi_p, aa=aa_legend, msc=msc_str)
                        else:
                            legend = "{m}{aa}".format(m=multi_p, aa=aa_legend)
                    if not self.display_only_aa_change:
                        # if patient.age_onset < 0:
                        #     self.free_slot(slot=slot, bottom_part=bottom_part)
                        #     continue
                        if patient.age_onset == -1:
                            legend = f"NA"
                        elif patient.age_onset == -2:
                            legend = "no seizure"
                        else:
                            legend = f"{patient.age_onset}"
                    if patient.NA_filtered_data:
                        color = self.na_filtered_data_color
                    else:
                        color = self.legend_mut_line_color
                    if not self.display_only_aa_change:
                        linewidth = 1
                    else:
                        if patient.NA_filtered_data:
                            linewidth = self.legend_mut_line_width + (np.log(nb_patients_na) / 2)
                        else:
                            linewidth = self.legend_mut_line_width + (np.log(nb_patients_known) / 2)

                    plt.plot([coord_mut[0], x_coord], [coord_mut[1], y_coord], c=color,
                             linewidth=linewidth, zorder=self.legend_mut_z_order, alpha=0.8,
                             linestyle=self.legend_mut_linestyle)

                    y_coord = self.legend_mut_y_coord_down if bottom_part \
                        else self.legend_mut_y_coord_up
                    va_option = "top" if bottom_part else "bottom"

                    # legend = patient.dna_change
                    color = self.legend_mut_color
                    if msc_in_the_place:
                        color = self.scatter_msc_color
                    if patient.NA_filtered_data:
                        color = self.na_filtered_data_color
                    with_bbox = False
                    if with_bbox:
                        plt.text(x=x_coord, y=y_coord, s=f'{legend}', ha='center', va=va_option, fontsize=6,
                                 rotation=-90,
                                 color="white", bbox=dict(facecolor=color, alpha=1, boxstyle="round", pad=0.2))
                    else:
                        plt.text(x=x_coord, y=y_coord, s=f'{legend}', ha='center', va=va_option, fontsize=6,
                                 rotation=-90,
                                 color=color)
                    if patient.NA_filtered_data:
                        index_count_NA += 1
                    else:
                        index_count += 1

    def get_max_variant_frequency(self, ):
        max_freq = 0
        for region in self.regions_list:
            if self.get_variants_freq() == 0:
                max_freq = max(max_freq, region.variants_frequency)
            else:
                # print(f"max_freq {max_freq}")
                # print(f"(region.variants_frequency / self.get_variants_freq()) "
                #       f"{(region.variants_frequency / self.get_variants_freq())}")
                max_freq = max((region.variants_frequency / self.get_variants_freq()), max_freq)
        return max_freq


    def plot_bar_variants_frequency(self, filter_option, max_variant_freq, holland_version=True, path_results=None,
                                    title_fig=None, show_fig=False):
        # fig, ax1 = plt.subplots(nrows=1, ncols=1,
        #                         figsize=(8, 8), num=f"{filter_option}: relative frequency of variants for each region")
        fig = plt.figure(figsize=(8, 8), num=f"{filter_option}: relative frequency of variants for each region")
        axe_plot = fig.subplots()

        reg_variants_freq = np.zeros(len(self.regions_list))
        x_ticks_labels = []
        bar_colors = []
        local_max_variant_freq = 0
        for i, region in enumerate(self.regions_list):
            bar_colors.append(region.color)
            if self.get_variants_freq() == 0:
                reg_variants_freq[i] = region.variants_frequency
            else:
                reg_variants_freq[i] = region.variants_frequency / self.get_variants_freq()
            local_max_variant_freq = np.max((reg_variants_freq[i], local_max_variant_freq))
            x_ticks_labels.append(region.region_id)

        if max_variant_freq < 0:
            max_variant_freq = local_max_variant_freq
        nb_regions = len(self.regions_list)
        x_pos = np.arange(0, nb_regions)
        linewidth = np.repeat(1, nb_regions)
        linewidth[reg_variants_freq == 0] = 0
        plt.bar(x_pos,
                reg_variants_freq, color=bar_colors,
                edgecolor=["black"] * nb_regions, linewidth=linewidth)
        for i, region in enumerate(self.regions_list):
            if reg_variants_freq[i] > 0:
                y = reg_variants_freq[i] / 2
                color = "grey"
                if holland_version:
                    color = "black"
                fontsize = 6
                if holland_version:
                    fontsize = 10
                plt.text(x=x_pos[i], y=y,
                         s=f"{region.nb_aa_with_mutation} / {region.nb_mutated_patients}",  # \n({region.nb_total_aa})",
                         color=color, zorder=22,
                         ha='center', va="center", fontsize=fontsize, fontweight='bold')

        # print(f"NaV1.6, relative frequency of variants: {np.round(self.get_variants_freq(), 4)}")
        # print(f"NaV1.6 nb_unique aa: {self.nb_unique_aa_mutated()}")
        # print(f"NaV1.6 nb aa: {self.nb_aa}")
        # properties = {'weight': 'bold'}
        plt.xticks(x_pos, x_ticks_labels, fontweight="bold")

        # self.axe_plot.axes.get_xaxis().set_visible(False)
        # self.axe_plot.axes.get_yaxis().set_visible(False)
        axe_plot.spines['bottom'].set_visible(False)
        axe_plot.spines['right'].set_visible(False)
        axe_plot.spines['top'].set_visible(False)
        axe_plot.spines['left'].set_visible(False)
        axe_plot.set_ylim(-0.05, max_variant_freq)

        y_pos = np.arange(0, int(np.ceil(max_variant_freq)))
        plt.yticks(y_pos, y_pos)
        # self.axe_plot.set_xlim(-1, self.total_width)

        # plt.xlabel('Epilepsy', fontweight="bold", fontsize=12)
        # plt.ylabel('Relative frequency of variants', fontweight="bold", fontsize=12)
        # plt.title(f"Nb aa mutated, unique ({self.nb_unique_aa_mutated()}), total ({self.nb_mutations()})")

        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        if holland_version:
            paper_inspiration = "Holland"
        else:
            paper_inspiration = "Zuberi"
        if title_fig is None:
            fig.savefig(f'{path_results}/{filter_option}_nav16_variant_frequency_{paper_inspiration}_{time_str}.pdf',
                        format="pdf")
        else:
            fig.savefig(f'{path_results}/{title_fig}_{time_str}.pdf',
                        format="pdf")
        if show_fig:
            plt.show()
        plt.close()

    def plot_element(self, plot_next=True, with_mutation_legends=True, filter_option=None, path_results=None,
                     show_plot=False, title_fig=None):
        if self.plotted:
            return
        self.plotted = True

        self.axe_plot = self.fig.subplots()

        self.plot_next_element()
        self.plot_mutations()
        self.plot_mutation_legends()
        for d in self.domains.values():
            d.plot_element()
        self.plot_legends()
        # plt.hlines(y=50, xmin=0, xmax=1000)

        self.axe_plot.axes.get_xaxis().set_visible(False)
        self.axe_plot.axes.get_yaxis().set_visible(False)
        self.axe_plot.spines['bottom'].set_visible(False)
        self.axe_plot.spines['right'].set_visible(False)
        self.axe_plot.spines['top'].set_visible(False)
        self.axe_plot.spines['left'].set_visible(False)
        self.axe_plot.set_ylim(0, self.total_height)
        self.axe_plot.set_xlim(-1, self.total_width)

        if filter_option is not None:
            plt.title(f"{filter_option}", y=1.10)

        # path_results = "/Users/pappyhammer/Documents/academique/SCN8A/python_results"
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

        if filter_option is None:
            if title_fig is not None:
                self.fig.savefig(f'{path_results}/{title_fig}_{time_str}.pdf',
                                 format="pdf")
            else:
                self.fig.savefig(f'{path_results}/nav16_fig_{time_str}.pdf',
                                 format="pdf")
            if show_plot:
                plt.show()
        else:
            if title_fig is not None:
                self.fig.savefig(f'{path_results}/{title_fig}_{time_str}.pdf',
                                 format="pdf")
            else:
                self.fig.savefig(f'{path_results}/nav16_fig_{filter_option}_{time_str}.pdf',
                                 format="pdf")
            if show_plot:
                plt.show()
        plt.close()

    def plot_legends(self):
        """
        plot some legends
        :return:
        """
        plt.text(x=-65, y=self.domains[1].segments[1].end_coord[1] + 7,
                 s="Extracellular\n     Space", color="black", zorder=15,
                 ha='left', va="center", fontsize=9)
        plt.text(x=-65, y=self.domains[1].segments[1].start_coord[1] - 7,
                 s="Cytoplasm", color="black", zorder=15,
                 ha='left', va="top", fontsize=9)


class Domain(CanalElement):
    def __init__(self, number, first_position, last_position, parent, start_coord, end_coord):
        super().__init__(first_position=first_position, last_position=last_position, no_elements_inside=False,
                         parent=parent, start_coord=start_coord, end_coord=end_coord)
        # 6 segments, from 1 to 6
        self.segments = dict()
        # domain number
        self.number = number
        self.next_domain = None
        self.previous_domain = None
        self.width = self.total_width / 4
        self.height = self.total_height * 0.9

    def set_next_domain(self, domain):
        self.next_domain = domain
        # self.set_next_element(canal_element=domain)

    def set_previous_domain(self, domain):
        self.previous_domain = domain
        # self.set_previous_element(canal_element=domain)

    def add_segment(self, segment):
        self.segments[segment.number] = segment
        # self.fill_elements_position(canal_element=segment)

    def plot_element(self, plot_next=True):
        if self.plotted:
            return

        self.plotted = True
        # print(f"{self} x: {self.start_coord[0] + (self.width/2)}, y: {self.start_coord[0]+(self.height*0.65)}")

        plt.text(x=self.start_coord[0] + (self.width * 0.36), y=self.start_coord[1] + (self.height * 0.9),
                 s=f"D{self.number}", color="black", zorder=15,
                 ha='center', va="center", fontsize=12, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        xy = (self.start_coord[0] + (self.width * 0.295), self.start_coord[1] + (self.height * 0.88))
        padding = 1.8
        width_segment = self.segments[1].width * 1.5
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment,
                                 height=padding * 2.5, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        # plotting the segment numbers
        if self.number == 1:
            letters_center = self.start_coord[1] + (self.height * 0.26)  # 0.16
            padding = 1.8
            border_up = letters_center + padding
            border_down = letters_center - padding
            for segment in self.segments.values():
                plt.text(x=segment.x_left_coord[0] + (segment.width * 0.5),
                         y=letters_center,
                         s=f"S{segment.number}", color="black", zorder=15,
                         ha='center', va="center", fontsize=8.5, fontweight='bold')
            # draw some cases for segments numbers
            # left vertical
            xy = (self.segments[1].x_left_coord[0], border_down)

            rect = patches.Rectangle(xy=xy,
                                     width=self.segments[6].x_right_coord[0] - self.segments[1].x_left_coord[0],
                                     height=padding * 2.3, fill=True, linestyle="solid", facecolor="white",
                                     edgecolor="black", zorder=15)
            axe_plot.add_patch(rect)
            # for s in np.arange(1, 7):
            #     plt.plot([self.segments[1].x_left_coord[0], self.segments[1].x_left_coord[0]],
            #          [border_up, border_down], linewidth=1, c="black")
            # # right vertical
            # plt.plot([self.segments[6].x_right_coord[0], self.segments[6].x_right_coord[0]],
            #          [border_up, border_down], linewidth=1, c="black")

    # use for debugging
    def is_it_filled(self):
        for i in np.arange(self.first_position, self.last_position + 1):
            if i not in self.elements_position:
                return False
        return True

    def __str__(self):
        return f"Domain {self.number}"


class Segment(CanalElement):
    def __init__(self, number, first_position, last_position, parent):
        super().__init__(first_position=first_position, last_position=last_position, parent=parent)
        self.width = int(self.total_width / 45)
        self.height = self.total_height / 3.5
        x_start = parent.start_coord[0] + (0.5 * self.width) + ((number - 1) * self.width)
        if number >= 5:
            x_start += self.width / 2
        if number == 6:
            x_start += self.width * 2
        y_start = (self.height * 1.2) + parent.start_coord[1]
        # start means the bottom part
        self.start_coord = (x_start, y_start)
        self.end_coord = (x_start, y_start + self.height)
        self.number = number
        self.voltage_sensor = True if (self.number == 4) else False

        # --------- plotting attributes  ----------
        # left_border
        self.x_left_coord = np.zeros(2)
        self.y_left_coord = np.zeros(2)
        self.x_left_coord[0] = self.start_coord[0] - self.width / 2
        self.x_left_coord[1] = self.x_left_coord[0]
        self.y_left_coord[0] = self.start_coord[1]
        self.y_left_coord[1] = self.end_coord[1]

        # rigth_border
        self.x_right_coord = np.zeros(2)
        self.y_right_coord = np.zeros(2)
        self.x_right_coord[0] = self.x_left_coord[0] + self.width
        self.x_right_coord[1] = self.x_right_coord[0]
        self.y_right_coord[0] = self.y_left_coord[0]
        self.y_right_coord[1] = self.y_left_coord[1]

        height_convex = (0.02 * self.height)
        # bottom part
        self.x_bottom_coord = np.zeros(5)
        self.y_bottom_coord = np.zeros(5)

        self.x_bottom_coord[0] = self.x_left_coord[0]
        self.x_bottom_coord[4] = self.x_right_coord[0]
        self.x_bottom_coord[2] = self.x_left_coord[0] + (0.5 * self.width)
        self.x_bottom_coord[1] = self.x_left_coord[0] + (0.25 * self.width)
        self.x_bottom_coord[3] = self.x_left_coord[0] + (0.75 * self.width)
        # to interpolate
        self.x_interp = np.linspace(self.x_bottom_coord[0], self.x_bottom_coord[4], num=20, endpoint=True)

        self.y_bottom_coord[0] = self.y_left_coord[0]
        self.y_bottom_coord[4] = self.y_left_coord[0]
        self.y_bottom_coord[2] = self.y_left_coord[0] - height_convex
        self.lowest_y_coord = self.y_bottom_coord[2]
        self.y_bottom_coord[1] = self.y_left_coord[0] - (height_convex * 0.8)
        self.y_bottom_coord[3] = self.y_left_coord[0] - (height_convex * 0.8)

        # top
        # b as bottom part of the top
        self.x_top_coord_b = np.zeros(5)
        self.y_top_coord_b = np.zeros(5)

        self.x_top_coord_b[0] = self.x_left_coord[1]
        self.x_top_coord_b[4] = self.x_right_coord[1]
        self.x_top_coord_b[2] = self.x_left_coord[1] + (0.5 * self.width)
        self.x_top_coord_b[1] = self.x_left_coord[1] + (0.25 * self.width)
        self.x_top_coord_b[3] = self.x_left_coord[1] + (0.75 * self.width)

        self.y_top_coord_b[0] = self.y_left_coord[1]
        self.y_top_coord_b[4] = self.y_left_coord[1]
        self.y_top_coord_b[2] = self.y_left_coord[1] - height_convex
        self.y_top_coord_b[1] = self.y_left_coord[1] - (height_convex * 0.8)
        self.y_top_coord_b[3] = self.y_left_coord[1] - (height_convex * 0.8)

        self.x_top_coord_t = np.zeros(5)
        self.y_top_coord_t = np.zeros(5)

        self.x_top_coord_t[0] = self.x_left_coord[1]
        self.x_top_coord_t[4] = self.x_right_coord[1]
        self.x_top_coord_t[2] = self.x_left_coord[1] + (0.5 * self.width)
        self.x_top_coord_t[1] = self.x_left_coord[1] + (0.25 * self.width)
        self.x_top_coord_t[3] = self.x_left_coord[1] + (0.75 * self.width)

        self.y_top_coord_t[0] = self.y_left_coord[1]
        self.y_top_coord_t[4] = self.y_left_coord[1]
        self.y_top_coord_t[2] = self.y_left_coord[1] + height_convex
        self.y_top_coord_t[1] = self.y_left_coord[1] + (height_convex * 0.8)
        self.y_top_coord_t[3] = self.y_left_coord[1] + (height_convex * 0.8)

        self.x_mut_coords = np.repeat(self.start_coord[0], self.len_interpol_mut)
        # check if number is even or odd, depending the start will be the end regarding the aa position
        if (self.number % 2) == 1:
            self.y_mut_coords = np.linspace(self.start_coord[1], self.end_coord[1],
                                            num=self.len_interpol_mut, endpoint=True)
        else:
            self.y_mut_coords = np.linspace(self.end_coord[1], self.start_coord[1],
                                            num=self.len_interpol_mut, endpoint=True)
            # print(f"{self.parent}, s {self.number}, y_mut_coords {self.y_mut_coords}")
        # --------- end plotting attributes  ----------

    # def get_coord_from_aa_position(self, aa_position):
    #     if aa_position < self.first_position or aa_position > self.last_position:
    #         return None
    #
    #     # distance percentage between first and last position
    #     perc = aa_position - self.first_position
    #     perc /= self.last_position-self.first_position
    #
    #     x_coord = self.start_coord[0]
    #
    #     y_coord = self.start_coord[1] + ((self.end_coord[1] - self.start_coord[1]) * perc)
    #
    #     return tuple([x_coord, y_coord])

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        # left_border

        axe_plot = self.get_older_parent().axe_plot

        plt.plot(self.x_left_coord, self.y_left_coord, c=self.segment_line_color)

        # rigth_border
        plt.plot(self.x_right_coord, self.y_right_coord, c=self.segment_line_color)

        # bottom part
        plt.plot(self.x_interp, CanalElement.interpolate_fct(self.x_bottom_coord, self.y_bottom_coord)(self.x_interp),
                 c=self.segment_line_color)

        # top
        plt.plot(self.x_interp, CanalElement.interpolate_fct(self.x_top_coord_b, self.y_top_coord_b)(self.x_interp),
                 c=self.segment_line_color)

        plt.plot(self.x_interp, CanalElement.interpolate_fct(self.x_top_coord_t, self.y_top_coord_t)(self.x_interp),
                 c=self.segment_line_color, zorder=2)

        if self.region is not None:
            x_interp = np.linspace(self.x_bottom_coord[0], self.x_bottom_coord[4], num=15, endpoint=True)
            y_interp = CanalElement.interpolate_fct(self.x_top_coord_t, self.y_top_coord_t)(x_interp)
            for i, x in enumerate(x_interp):
                plt.plot([x, x], [self.y_top_coord_t[0], y_interp[i]], linewidth=1, c=self.region.color, zorder=1)
            for i in np.arange(len(y_interp) // 2):
                plt.plot([x_interp[i], x_interp[-i]], [y_interp[i], y_interp[-i]], linewidth=1,
                         c=self.region.color, zorder=1)

            y_interp = CanalElement.interpolate_fct(self.x_bottom_coord, self.y_bottom_coord)(x_interp)
            for i, x in enumerate(x_interp):
                plt.plot([x, x], [self.y_bottom_coord[0], y_interp[i]], linewidth=1, c=self.region.color, zorder=1)
            for i in np.arange(len(y_interp) // 2):
                plt.plot([x_interp[i], x_interp[-i]], [y_interp[i], y_interp[-i]], linewidth=1,
                         c=self.region.color, zorder=1)
            # shift = 0.3
            # self.y_top_coord_t[1] -= shift
            # self.y_top_coord_t[2] -= 0.2
            # self.y_top_coord_t[3] -= shift
            # plt.plot(self.x_interp, CanalElement.interpolate_fct(self.x_top_coord_t, self.y_top_coord_t)(self.x_interp),
            #          linewidth=4, c=self.region.color, zorder=1)

            xy = (self.x_left_coord[0], self.y_left_coord[0])
            rect = patches.Rectangle(xy=xy,
                                     width=self.x_right_coord[0] - self.x_left_coord[0],
                                     height=self.height, fill=True,
                                     facecolor=self.region.color, zorder=1)
            axe_plot.add_patch(rect)

        # --------- add + + + +  to the 4th segment ---------
        plot_plus_segment_4 = False
        if plot_plus_segment_4:
            if self.number == 4:
                nb_plus = 5
                beg = int(self.len_interpol_mut / 15)
                inter = int((self.len_interpol_mut - (beg * 2)) / nb_plus)
                index_coord_plus = np.arange(self.len_interpol_mut)[int(beg * 2.5)::inter]
                for i in np.arange(nb_plus):
                    # print(f"self.x_mut_coords[index_coord_plus[i]] {self.x_mut_coords[index_coord_plus[i]]}")
                    # print(f"self.y_mut_coords[index_coord_plus[i]] {self.y_mut_coords[index_coord_plus[i]]}")
                    plt.scatter(self.x_mut_coords[index_coord_plus[i]], self.y_mut_coords[index_coord_plus[i]],
                                marker='+',
                                color="black",
                                linewidth=1.5, s=60, zorder=3, alpha=1)
        # --------- end to add + + + +  to the 4th segment ---------

        # --------- draw mb wall -----------
        wall_color = "black"
        wall_line_width = 0.7
        wall_filled_color = "lightgrey"
        len_limit_wall = 20
        just_lines = False

        if self.number == 1:
            if self.parent.number == 1:

                if just_lines:
                    plt.hlines(y=self.start_coord[1], xmin=self.x_left_coord[0] - len_limit_wall,
                               xmax=self.x_left_coord[0],
                               linewidth=wall_line_width,
                               color=wall_color)
                    plt.hlines(y=self.end_coord[1], xmin=self.x_left_coord[0] - len_limit_wall,
                               xmax=self.x_left_coord[0],
                               linewidth=wall_line_width,
                               color=wall_color)
                else:
                    xy = ((self.x_left_coord[0] - len_limit_wall), self.start_coord[1])
                    rect = patches.Rectangle(xy=xy,
                                             width=len_limit_wall, height=self.height, fill=True,
                                             facecolor=wall_filled_color,
                                             edgecolor=wall_color, zorder=1)
                    axe_plot.add_patch(rect)
            else:
                previous_segment = self.parent.previous_domain.segments[6]
                if just_lines:
                    plt.hlines(y=self.start_coord[1], xmin=previous_segment.x_right_coord[0],
                               xmax=self.x_left_coord[0], linewidth=wall_line_width,
                               color=wall_color)
                    plt.hlines(y=self.end_coord[1], xmin=previous_segment.x_right_coord[0],
                               xmax=self.x_left_coord[0], linewidth=wall_line_width,
                               color=wall_color)
                else:
                    xy = (previous_segment.x_right_coord[0], self.start_coord[1])
                    width_rect = self.x_left_coord[0] - previous_segment.x_right_coord[0]
                    rect = patches.Rectangle(xy=xy,
                                             width=width_rect, height=self.height, fill=True,
                                             facecolor=wall_filled_color,
                                             edgecolor=wall_color, zorder=1)
                    axe_plot.add_patch(rect)
        if (self.number == 6) and (self.parent.number == 4):
            # drawing wall after and before
            if just_lines:
                plt.hlines(y=self.start_coord[1], xmin=self.x_right_coord[0],
                           xmax=self.x_right_coord[0] + len_limit_wall, linewidth=wall_line_width,
                           color=wall_color)
                plt.hlines(y=self.end_coord[1], xmin=self.x_right_coord[0],
                           xmax=self.x_right_coord[0] + len_limit_wall, linewidth=wall_line_width,
                           color=wall_color)
            else:
                xy = (self.x_right_coord[0], self.start_coord[1])
                width_rect = len_limit_wall
                rect = patches.Rectangle(xy=xy,
                                         width=width_rect, height=self.height, fill=True,
                                         facecolor=wall_filled_color,
                                         edgecolor=wall_color, zorder=1)
                axe_plot.add_patch(rect)
        if (self.number == 5) or (self.number == 6):
            previous_segment = self.parent.segments[self.number - 1]
            if just_lines:
                plt.hlines(y=self.start_coord[1], xmin=previous_segment.x_right_coord[0],
                           xmax=self.x_left_coord[0], linewidth=wall_line_width,
                           color=wall_color)
                plt.hlines(y=self.end_coord[1], xmin=previous_segment.x_right_coord[0],
                           xmax=self.x_left_coord[0], linewidth=wall_line_width,
                           color=wall_color)
            else:
                xy = (previous_segment.x_right_coord[0], self.start_coord[1])
                width_rect = self.x_left_coord[0] - previous_segment.x_right_coord[0]
                rect = patches.Rectangle(xy=xy,
                                         width=width_rect, height=self.height, fill=True,
                                         facecolor=wall_filled_color,
                                         edgecolor=wall_color, zorder=1)
                axe_plot.add_patch(rect)
        # -------- end wall -----------

        # print(f"Element {self} plotted, go to next")
        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return f"Segment {self.number} from {str(self.parent)}"


class Loop(CanalElement):
    def __init__(self, first_position, last_position, parent, previous_element=None, next_element=None):
        super().__init__(first_position=first_position, last_position=last_position, previous_element=previous_element,
                         next_element=next_element, parent=parent)
        # setting the links with previous and next element
        previous_element.set_next_element(self)
        next_element.set_previous_element(self)


class PoreLoop(Loop):
    def __init__(self, first_position, last_position, intra_mb_first_position, intra_mb_last_position, parent,
                 previous_element=None, next_element=None):
        super().__init__(first_position=first_position, last_position=last_position,
                         previous_element=previous_element, next_element=next_element, parent=parent)
        self.intra_mb_first_position = intra_mb_first_position
        self.intra_mb_last_position = intra_mb_last_position

        # -------- plotting attributes --------
        self.x_mut_coords = None
        self.y_mut_coords = None

        self.start_coord = (previous_element.start_coord[0], previous_element.end_coord[1])
        self.width = next_element.start_coord[0] - previous_element.start_coord[0]

        self.x_mut_coords = np.linspace(previous_element.start_coord[0], next_element.start_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(16)
        y_coord = np.zeros(16)

        x_coord[0] = previous_element.start_coord[0]
        # x_coord[12] = next_element.start_coord[0]
        segment_width = previous_element.width
        # self.width = x_coord[14] - x_coord[0]

        # first loop
        x_coord[1] = x_coord[0] + (0.2 * segment_width)
        x_coord[2] = x_coord[0] + (0.35 * segment_width)
        x_coord[3] = x_coord[0] + (0.6 * segment_width)
        x_coord[4] = x_coord[0] + (0.8 * segment_width)
        x_coord[5] = x_coord[0] + segment_width

        # 2nd loop
        x_coord[6] = x_coord[5] + (0.15 * segment_width)
        x_coord[7] = x_coord[5] + (0.3 * segment_width)
        x_coord[8] = x_coord[5] + (0.7 * segment_width)
        x_coord[9] = x_coord[5] + (0.85 * segment_width)
        x_coord[10] = x_coord[5] + segment_width

        # 3rd and last loop
        x_coord[11] = x_coord[10] + (0.1 * segment_width)
        x_coord[12] = x_coord[10] + (0.25 * segment_width)
        x_coord[13] = x_coord[10] + (0.7 * segment_width)
        x_coord[14] = x_coord[10] + (0.85 * segment_width)
        x_coord[15] = x_coord[10] + segment_width

        y_coord[0] = previous_element.end_coord[1]
        height_segment = previous_element.height
        self.first_loop_height = height_segment * 0.4
        second_loop_height = height_segment * 0.45
        third_loop_height = height_segment * 0.2
        # first loop
        y_coord[1] = y_coord[0] + (self.first_loop_height * 0.5)
        y_coord[2] = y_coord[0] + (self.first_loop_height * 0.8)
        y_coord[3] = y_coord[0] + (self.first_loop_height * 0.8)
        y_coord[4] = y_coord[0] + (self.first_loop_height * 0.5)
        y_coord[5] = y_coord[0]

        # second loop
        y_coord[6] = y_coord[0] - (second_loop_height * 0.5)
        y_coord[7] = y_coord[0] - (second_loop_height * 0.7)
        y_coord[8] = y_coord[0] - (second_loop_height * 0.7)
        y_coord[9] = y_coord[0] - (second_loop_height * 0.5)
        y_coord[10] = y_coord[0]

        # third loop
        y_coord[11] = y_coord[0] + (third_loop_height * 0.5)
        y_coord[12] = y_coord[0] + (third_loop_height * 0.8)
        y_coord[13] = y_coord[0] + (third_loop_height * 0.8)
        y_coord[14] = y_coord[0] + (third_loop_height * 0.5)
        y_coord[15] = y_coord[0]

        # y_coord[15] = next_element.end_coord[1]

        # print(f'x_coord {x_coord}')
        # print(f'y_coord {y_coord}')
        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True

        if self.region is None:
            color = self.short_loop_line_color
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, c=color,
                 linewidth=self.short_loop_line_width)

        plt.text(x=self.start_coord[0] + (self.width / 2) + 1.8, y=self.start_coord[1] +
                                                                   (self.first_loop_height * 0.3) + 1,
                 s="P", color="black", zorder=15,
                 ha='center', va="center", fontsize=10, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        width_segment = self.previous_element.width * 0.73
        xy = (self.start_coord[0] + (self.width / 2) + 2.3 - (width_segment / 2),
              self.start_coord[1] + (self.first_loop_height * 0.3) + 1 - 1.3)
        padding = 1
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment - 0.5,
                                 height=padding * 3.5, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return f"Pore loop from {str(self.parent)}"


class LargeLoop(Loop):
    def __init__(self, number, first_position, last_position, parent,
                 previous_domain, next_domain):
        super().__init__(first_position=first_position, last_position=last_position,
                         previous_element=previous_domain.segments[6], next_element=next_domain.segments[1],
                         parent=parent)
        self.number = number
        self.previous_domain = previous_domain
        self.next_domain = next_domain

        # -------- plotting attributes --------
        self.x_mut_coords = None
        self.y_mut_coords = None

        self.x_mut_coords = np.linspace(self.previous_element.start_coord[0], self.next_element.start_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(8)
        y_coord = np.zeros(8)

        x_coord[0] = self.previous_element.start_coord[0]
        x_coord[7] = self.next_element.start_coord[0]
        self.width = x_coord[7] - x_coord[0]
        x_coord[1] = x_coord[0] + (0.2 * self.width)
        x_coord[2] = x_coord[0] + (0.3 * self.width)
        x_coord[3] = x_coord[0] + (0.4 * self.width)
        x_coord[4] = x_coord[0] + (0.6 * self.width)
        x_coord[5] = x_coord[0] + (0.7 * self.width)
        x_coord[6] = x_coord[0] + (0.8 * self.width)

        self.height = (1 * self.previous_element.height)
        if number == 2:
            self.height = (0.9 * self.previous_element.height)
        y_coord[0] = self.next_element.lowest_y_coord
        y_coord[7] = y_coord[0]
        sign_change = -1
        y_coord[1] = y_coord[0] + ((self.height * 0.3) * sign_change)
        y_coord[2] = y_coord[0] + ((self.height * 0.5) * sign_change)
        y_coord[3] = y_coord[0] + ((self.height * 0.6) * sign_change)
        y_coord[4] = y_coord[0] + ((self.height * 0.6) * sign_change)
        y_coord[5] = y_coord[0] + ((self.height * 0.5) * sign_change)
        y_coord[6] = y_coord[0] + ((self.height * 0.3) * sign_change)

        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        if self.region is None:
            color = self.short_loop_line_color
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, c=color,
                 linewidth=self.short_loop_line_width)

        plt.text(x=self.previous_element.start_coord[0] + (0.5 * self.width),
                 y=self.next_element.lowest_y_coord - (0.2 * self.height),
                 s=f"L{self.number}", color="black", zorder=15,
                 ha='center', va="center", fontsize=10, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        width_segment = self.previous_element.width * 1.1
        xy = (self.previous_element.start_coord[0] + (0.5 * self.width) - (width_segment / 2),
              self.next_element.lowest_y_coord - (0.2 * self.height) - 1.5)
        padding = 1
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment,
                                 height=padding * 3.7, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return f"Large loop {self.number}"


class InactivationGateLoop(Loop):
    def __init__(self, first_position, last_position, parent,
                 previous_domain, next_domain):
        super().__init__(first_position=first_position, last_position=last_position,
                         previous_element=previous_domain.segments[6], next_element=next_domain.segments[1],
                         parent=parent)
        self.previous_domain = previous_domain
        self.next_domain = next_domain

        # -------- plotting attributes --------
        self.x_mut_coords = None
        self.y_mut_coords = None

        self.x_mut_coords = np.linspace(self.previous_element.start_coord[0], self.next_element.start_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(6)
        y_coord = np.zeros(6)

        x_coord[0] = self.previous_element.start_coord[0]
        x_coord[5] = self.next_element.start_coord[0]
        self.width = x_coord[5] - x_coord[0]
        x_coord[2] = x_coord[0] + (0.35 * self.width)
        x_coord[3] = x_coord[0] + (0.65 * self.width)
        x_coord[1] = x_coord[0] + (0.2 * self.width)
        x_coord[4] = x_coord[0] + (0.8 * self.width)

        self.height = (0.3 * self.previous_element.height)
        y_coord[0] = self.next_element.lowest_y_coord
        y_coord[5] = y_coord[0]
        sign_change = -1
        y_coord[1] = y_coord[0] + ((self.height * 0.5) * sign_change)
        y_coord[2] = y_coord[0] + ((self.height * 0.8) * sign_change)
        y_coord[3] = y_coord[0] + ((self.height * 0.8) * sign_change)
        y_coord[4] = y_coord[0] + ((self.height * 0.5) * sign_change)

        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        if self.region is None:
            color = self.short_loop_line_color
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, color=color,
                 linewidth=self.short_loop_line_width)

        plt.text(x=self.previous_element.start_coord[0] + (0.5 * self.width),
                 y=self.next_element.lowest_y_coord - (0.35 * self.height),
                 s=f"L3", color="black", zorder=15,
                 ha='center', va="center", fontsize=10, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        width_segment = self.previous_element.width * 1.1
        xy = (self.previous_element.start_coord[0] + (0.5 * self.width) - (width_segment / 2),
              self.next_element.lowest_y_coord - (0.35 * self.height) - 1.5)
        padding = 1
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment,
                                 height=padding * 3.7, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        plt.text(x=self.previous_element.start_coord[0] + (0.5 * self.width),
                 y=self.next_element.lowest_y_coord - (1.4 * self.height) - 1.5,
                 s="Inactivation\n  Gate", color="black", zorder=100,
                 ha='center', va="center", fontsize=8, fontweight='bold')

        # draw some cases for segments numbers
        # left vertical
        axe_plot = self.get_older_parent().axe_plot
        xy = (self.previous_element.start_coord[0] * 0.99,
              self.next_element.lowest_y_coord - (1.75 * self.height) - 1.5)
        padding = 2
        rect = patches.Rectangle(xy=xy,
                                 width=self.width * 1.2, linewidth=1,
                                 height=padding * 3.2, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        # solid
        axe_plot.add_patch(rect)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return "Inactivation gate loop"


class ShortLoop(Loop):
    def __init__(self, number, first_position, last_position, parent,
                 previous_element=None, next_element=None):
        super().__init__(first_position=first_position, last_position=last_position,
                         previous_element=previous_element, next_element=next_element, parent=parent)
        self.number = number

        # if number if even, then the loop is on the bottom
        self.is_even = ((self.number % 2) == 0)

        # -------- plotting attributes --------
        self.x_mut_coords = None
        self.y_mut_coords = None

        self.x_mut_coords = np.linspace(previous_element.start_coord[0], next_element.start_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(6)
        y_coord = np.zeros(6)

        x_coord[0] = previous_element.start_coord[0]
        x_coord[5] = next_element.start_coord[0]
        self.width = x_coord[5] - x_coord[0]
        x_coord[2] = x_coord[0] + (0.35 * self.width)
        x_coord[3] = x_coord[0] + (0.65 * self.width)
        x_coord[1] = x_coord[0] + (0.2 * self.width)
        x_coord[4] = x_coord[0] + (0.8 * self.width)

        self.height = (0.15 * previous_element.height)
        if self.is_even:
            y_coord[0] = next_element.lowest_y_coord
        else:
            y_coord[0] = previous_element.end_coord[1]
        y_coord[5] = y_coord[0]
        sign_change = -1 if self.is_even else 1
        y_coord[1] = y_coord[0] + ((self.height * 0.5) * sign_change)
        y_coord[2] = y_coord[0] + ((self.height * 0.8) * sign_change)
        y_coord[3] = y_coord[0] + ((self.height * 0.8) * sign_change)
        y_coord[4] = y_coord[0] + ((self.height * 0.5) * sign_change)

        # print(f"x_coord {x_coord}, y_coord {y_coord}, previous_element {previous_element},"
        #       f"next_element {next_element}"
        #       f" previous_element.start_coord[0] {previous_element.start_coord[0]}, "
        #       f"next_element.start_coord[0] {next_element.start_coord[0]}")
        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        if self.region is None:
            color = self.short_loop_line_color
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, c=color,
                 linewidth=self.short_loop_line_width)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return f"Short loop {self.number} from {str(self.parent)}"


class NTerminal(CanalElement):
    def __init__(self, first_position, last_position, domain, parent):
        super().__init__(first_position=first_position, last_position=last_position, parent=parent)
        self.domain = domain
        self.domain.set_previous_element(canal_element=self)
        segment = domain.segments[1]
        self.set_next_element(canal_element=segment)
        self.height = segment.height * 0.25
        self.start_coord = (16, segment.start_coord[1] - (self.height / 2))
        self.end_coord = (segment.start_coord[0], segment.lowest_y_coord)
        self.width = self.end_coord[0] - self.start_coord[0]

        # -------- plotting attributes --------
        self.x_mut_coords = None
        self.y_mut_coords = None

        self.x_mut_coords = np.linspace(self.start_coord[0], self.end_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(5)
        y_coord = np.zeros(5)

        x_coord[4] = self.end_coord[0]
        x_coord[3] = x_coord[4] - (0.25 * self.width)
        x_coord[2] = x_coord[4] - (0.4 * self.width)
        x_coord[1] = x_coord[4] - (0.6 * self.width)
        x_coord[0] = x_coord[4] - self.width

        y_coord[4] = self.end_coord[1]
        y_coord[3] = y_coord[4] - (self.height * 0.5)
        y_coord[2] = y_coord[4] - (self.height * 0.75)
        y_coord[1] = y_coord[4] - (self.height * 0.75)
        y_coord[0] = y_coord[4] - (self.height * 0.4)

        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        if self.region is None:
            color = self.short_loop_line_width
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, color=color,
                 linewidth=self.short_loop_line_width)

        plt.text(x=self.start_coord[0] - 8.5, y=self.start_coord[1], s="N", color="black", zorder=15,
                 ha='center', va="center", fontsize=10, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        width_segment = self.domain.segments[1].width * 0.73
        xy = (self.start_coord[0] - 8.5 - (width_segment / 2), self.start_coord[1] - 1.3)
        padding = 1
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment,
                                 height=padding * 3.5, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return "N-Terminal"


class CTerminal(CanalElement):
    def __init__(self, first_position, last_position, domain, parent):
        super().__init__(first_position=first_position, last_position=last_position, parent=parent)
        self.domain = domain
        self.domain.set_next_element(canal_element=self)
        self.set_previous_element(canal_element=domain.segments[6])
        domain.segments[6].set_next_element(self)
        last_segment = self.domain.segments[6]

        # -------- plotting attributes --------
        self.start_coord = (last_segment.start_coord[0], last_segment.lowest_y_coord)
        self.end_coord = (domain.segments[6].end_coord[0] + (domain.segments[6].width * 2.5),
                          self.get_older_parent().start_coord[1] + 15)

        self.x_mut_coords = None
        self.y_mut_coords = None

        self.x_mut_coords = np.linspace(self.start_coord[0], self.end_coord[0],
                                        num=self.len_interpol_mut, endpoint=True)

        x_coord = np.zeros(8)
        y_coord = np.zeros(8)

        x_coord[0] = self.start_coord[0]
        self.width = self.end_coord[0] - self.start_coord[0]
        x_coord[1] = x_coord[0] + (0.2 * self.width)
        x_coord[2] = x_coord[0] + (0.4 * self.width)
        x_coord[3] = x_coord[0] + (0.6 * self.width)
        x_coord[4] = x_coord[0] + (0.75 * self.width)
        x_coord[5] = x_coord[0] + (0.8 * self.width)
        x_coord[6] = x_coord[0] + (0.9 * self.width)
        x_coord[7] = self.end_coord[0]

        self.height = self.start_coord[1] - self.end_coord[1]

        y_coord[0] = self.start_coord[1]
        y_coord[1] = y_coord[0] - (self.height * 0.1)
        y_coord[2] = y_coord[0] - (self.height * 0.2)
        y_coord[3] = y_coord[0] - (self.height * 0.3)
        y_coord[4] = y_coord[0] - (self.height * 0.4)
        y_coord[5] = y_coord[0] - (self.height * 0.5)
        y_coord[6] = y_coord[0] - (self.height * 0.8)
        y_coord[7] = self.end_coord[1]

        # print(f"x_coord {x_coord}, y_coord {y_coord}")

        self.y_mut_coords = CanalElement.interpolate_fct(x_coord, y_coord)(self.x_mut_coords)

        # self.set_loop_coordinates()
        # -------- end plotting attributes --------

    def plot_element(self, plot_next=True):
        if self.plotted:
            return
        self.plotted = True
        if self.region is None:
            color = self.short_loop_line_color
        else:
            color = self.region.color
        plt.plot(self.x_mut_coords, self.y_mut_coords, c=color,
                 linewidth=self.short_loop_line_width)

        plt.text(x=self.end_coord[0] + 12, y=self.end_coord[1], s="C", color="black", zorder=15,
                 ha='center', va="center", fontsize=10, fontweight='bold')

        axe_plot = self.get_older_parent().axe_plot
        width_segment = self.domain.segments[6].width * 0.73
        xy = (self.end_coord[0] + 12 - (width_segment / 2), self.end_coord[1] - 1.3)
        padding = 1
        rect = patches.Rectangle(xy=xy,
                                 width=width_segment,
                                 height=padding * 3.5, fill=True, linestyle="solid", facecolor="white",
                                 edgecolor="black", zorder=15)
        axe_plot.add_patch(rect)

        super().plot_element(plot_next=plot_next)

    def __str__(self):
        return "C-Terminal"


def get_aa_change_position(scn8a_df):
    """
    Set the field 'aa.change.position' in scn8a_df based on "aa.change"
    :param scn8a_df:
    :return:
    """
    scn8a_df['aa.change.position'] = scn8a_df['aa.change']
    for n, aa_change in scn8a_df['aa.change'].items():
        aa_change = str(aa_change)
        # removing the spaces
        aa_change = aa_change.replace(" ", "")

        if aa_change == "nan":
            # print(f"{n} NaN")
            continue
        # for non-sens mutation, finishing by *
        if aa_change[-1] == "*":
            scn8a_df.at[n, 'aa.change.position'] = int(aa_change[3:-1])
            continue
        if (len(aa_change) > 3) and (aa_change[-3:] == "del"):
            # format should be Pro1428_Lys1473del
            scn8a_df.at[n, 'aa.change.position'] = int(aa_change[3:7])
            continue
        if (len(aa_change) > 5) and (aa_change[:6] == "splice"):
            # special for patient with splice mutation and not aa change identify
            scn8a_df.at[n, 'aa.change.position'] = int(aa_change[6:])
            scn8a_df.at[n, 'aa.change'] = scn8a_df.at[n, 'dna.change']
            # print(f'splice mutation: {scn8a_df["aa.change"]}')
            continue
        if not (7 <= len(aa_change) <= 10):
            scn8a_df.at[n, 'aa.change.position'] = np.NaN
            # print(f"{n} not the good len {aa_change}")
            continue

        scn8a_df.at[n, 'aa.change.position'] = int(aa_change[3:-3])


def generate_fig1_A(scn8a_df, path_results):
    generate_canal_fig_with_all_patients(scn8a_df, path_results, holland_version=True, title_fig="fig1A")


def generate_canal_fig_with_all_patients(scn8a_df, path_results, holland_version, title_fig):
    """
    Will produce fig1 A, representing the canal and location of each variant over it.
    :param scn8a_df:
    :param path_results:
    :param holland_version: if True eans we use region as define in Holland et al., otherwise as Zuberi et al.
    :return:
    """

    # patients with all mutation functions as well as patients without seizure

    patient_mutation_dict = dict()

    for n, patient_series in scn8a_df.iterrows():
        patient = SCN8APatient(patient_series)
        # keeping only patients from our cohort
        if not patient.msc:
            continue

        if patient.aa_change_position not in patient_mutation_dict:
            patient_mutation_dict[patient.aa_change_position] = []
        patient_mutation_dict[patient.aa_change_position].append(patient)

    fig = plt.figure(figsize=(12, 6))
    nav16 = Nav16(fig=fig)
    nav16.add_mutations(patient_mutation_dict)

    nav16.build_regions(holland_version=holland_version)

    get_some_stat = False
    if get_some_stat:
        age_group_1a = []
        age_group_1b = []
        age_group_2 = []
        for patients in patient_mutation_dict.values():
            for patient in patients:
                if patient.msc:
                    print(f"id {patient.patient_id}, dna_change {patient.dna_change}, "
                          f"one_letter {patient.aa_change_one_letter}, region: {patient.region.region_id}")
                    if patient.group_1_a:
                        age_group_1a.append(patient.age_onset)
                    elif patient.group_1_b:
                        age_group_1b.append(patient.age_onset)
                    elif patient.group_2:
                        age_group_2.append(patient.age_onset)
        print(f"group 1a: min: {np.min(age_group_1a)}, max: {np.max(age_group_1a)}, all: {age_group_1a}")
        print(f"group 1b: min: {np.min(age_group_1b)}, max: {np.max(age_group_1b)}, all: {age_group_1b}")
        print(f"group 2: min: {np.min(age_group_2)}, max: {np.max(age_group_2)}, all: {age_group_2}")

    nav16.plot_element(path_results=path_results, show_plot=False, title_fig=title_fig)
    plt.close()


def generate_fig1_B_to_D(scn8a_df, path_results):
    """
    Will produce fig1 B to F, representing the distribution of variants, according ot Holland  region
    :param scn8a_df:
    :param path_results:
    :return:
    """
    # means we use region as define in Holland et al.
    holland_version = True

    # we include only missence and epilepsy_related variants
    only_missence = True
    only_epilepsy = True

    filters_for_fig1_B_C_D = ["group 1a", "group 1b", "group 2"]
    # keeping only patients from our cohort
    only_msc = True

    # used to get the same scale
    max_variant_freq = 0

    # ___________ ----- A first loop to get the max frequency for each variant
    for filter_option in filters_for_fig1_B_C_D:
        patient_mutation_dict = dict()

        for n, patient_series in scn8a_df.iterrows():
            patient = SCN8APatient(patient_series)
            if not keep_patient(patient=patient, filter_option=filter_option, only_msc=only_msc,
                                only_missence=only_missence, only_epilepsy=only_epilepsy):
                    continue

            if patient.aa_change_position not in patient_mutation_dict:
                patient_mutation_dict[patient.aa_change_position] = []
            patient_mutation_dict[patient.aa_change_position].append(patient)

        fig = plt.figure(figsize=(12, 6))
        nav16 = Nav16(fig=fig)
        nav16.add_mutations(patient_mutation_dict)

        nav16.build_regions(holland_version=holland_version)
        max_variant_freq = max(nav16.get_max_variant_frequency(), max_variant_freq)
        plt.close()

    for filter_option in filters_for_fig1_B_C_D:
        # keep the SCN8APatient instance. The key is an int representing the aa position and the value
        # is a list of SCN8APatient
        patient_mutation_dict = dict()

        for n, patient_series in scn8a_df.iterrows():
            patient = SCN8APatient(patient_series)
            if not keep_patient(patient=patient, filter_option=filter_option, only_msc=only_msc,
                                only_missence=only_missence, only_epilepsy=only_epilepsy):
                continue
            if patient.aa_change_position not in patient_mutation_dict:
                patient_mutation_dict[patient.aa_change_position] = []
            patient_mutation_dict[patient.aa_change_position].append(patient)

        fig = plt.figure(figsize=(12, 6))
        nav16 = Nav16(fig=fig)
        nav16.add_mutations(patient_mutation_dict)

        # build the region, attributing for each element of the canal a region
        regions = nav16.build_regions(holland_version=holland_version)

        if filter_option == "group 1a":
            title_fig = "Fig1B"
        elif filter_option == "group 1b":
            title_fig = "Fig1C"
        else:
            title_fig = "Fig1D"
        nav16.plot_bar_variants_frequency(filter_option=filter_option, holland_version=holland_version,
                                          path_results=path_results, max_variant_freq=max_variant_freq,
                                          title_fig=title_fig)
        plt.close()

    # then fig1C with all patients

    only_msc = False
    patient_mutation_dict = dict()

    for n, patient_series in scn8a_df.iterrows():
        patient = SCN8APatient(patient_series)
        if not keep_patient(patient=patient, filter_option=None, only_msc=only_msc,
                            only_missence=only_missence, only_epilepsy=only_epilepsy):
            continue
        if patient.aa_change_position not in patient_mutation_dict:
            patient_mutation_dict[patient.aa_change_position] = []
        patient_mutation_dict[patient.aa_change_position].append(patient)

    plt.close()


def main():
    scn8a_df = pd.read_csv("/Users/pappyhammer/Documents/academique/SCN8A/github/SCN8A/data/SCN8A.patients.csv")
    total_nb_patients = scn8a_df.shape[0]

    path_results = "/Users/pappyhammer/Documents/academique/SCN8A/python_results/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results += f"{time_str}/"
    os.mkdir(path_results)

    # defining the amino acid position of each variant when available
    get_aa_change_position(scn8a_df=scn8a_df)

    # remove NaN values
    scn8a_df = scn8a_df[scn8a_df['aa.change.position'].notnull()]

    # generating all figures from the paper
    generate_fig1_A(scn8a_df, path_results)
    generate_fig1_B_to_D(scn8a_df, path_results)



def misc_and_debug_method(scn8a_df, path_results):
    """
    Miscellaneous operations, used for debugging and generating first version of the figures
    :param scn8a_df:
    :param path_results:
    :return:
    """
    filter_options = ["group 1", "group 1a", "group 1b", "group 2", "spasms", "mj",
                      "tonic seizures", "GTCS", "convulsive status", "focal seizure", "Focal seizure -> generalized",
                      "febrile seizure", "other type of seizure", "no seizure",
                      "first EEG known",
                      "normal EEG", "Non paroxysmal abnormality EEG", "focal paroxysmal abnormality EEG",
                      "multifocal paroxysmal abnormality EEG", "hypsarrythmia", "no worries on EEG",
                      "non paroxysmal group EEG", "paroxysmal abnormality EEG",
                      "SCB seizure free", "SCB non seizure free", "SCB worsening", "SCB seizure reduction",
                      "SCB no effect",
                      "SCB no effect or worsening",
                      "sudep", "AED seizure control", "AED no seizure control", "AED partial seizure control",
                      "normal development after onset", "delayed development after onset",
                      "delayed development without regression after onset",
                      "normal development before onset", "delayed development before onset",
                      "delayed development with regression after onset",
                      "onset before 3 months", "onset after 3 months", "onset before 4 months", "onset after 4 months",
                      "onset before 1 month", "onset after 1 month",
                      "onset before 1 month with AED info", "onset after 1 month with AED info",
                      "onset before 3 months with AED info", "onset after 3 months with AED info",
                      "epilepsy_missense", "PHT_seizure_free", "CBZ_seizure_free", "OXC_seizure_free",
                      "LTG_seizure_free", "LCM_seizure_free", "TPM_seizure_free", "ZON_seizure_free",
                      "PHT_non_seizure_free", "CBZ_non_seizure_free", "OXC_non_seizure_free",
                      "LTG_non_seizure_free", "LCM_non_seizure_free", "TPM_non_seizure_free", "ZON_non_seizure_free",
                      "PHT_no_effect", "CBZ_no_effect", "OXC_no_effect",
                      "LTG_no_effect", "LCM_no_effect", "TPM_no_effect", "ZON_no_effect",
                      "PHT_worsening", "CBZ_worsening", "OXC_worsening",
                      "LTG_worsening", "LCM_worsening", "TPM_worsening", "ZON_worsening",
                      "PHT_no_effect_or_worsening", "CBZ_no_effect_or_worsening", "OXC_no_effect_or_worsening",
                      "LTG_no_effect_or_worsening", "LCM_no_effect_or_worsening", "TPM_no_effect_or_worsening",
                      "ZON_no_effect_or_worsening",
                      "PHT_seizure_reduction", "CBZ_seizure_reduction", "OXC_seizure_reduction",
                      "LTG_seizure_reduction", "LCM_seizure_reduction", "TPM_seizure_reduction",
                      "ZON_seizure_reduction",
                      None]
    filters_all = filter_options
    # ofc.birth	ofc.evolution
    filters_for_ofc = ["ofc birth micro", "ofc birth normal", "ofc birth macro", "ofc evolution micro",
                       "ofc evolution normal", "ofc evolution macro", "ofc evolution downward"]
    filters_for_mri = ["mri onset normal", "mri onset abnormal", "mri follow-up normal", "mri follow-up abnormal"]
    filters_for_paper_fig1 = ["group 1a", "group 1b", "group 2", None]
    filters_for_paper_fig1_suppl = ["group 1", "group 2", None]
    filters_for_paper_fig3 = ["PHT_seizure_free", "CBZ_seizure_free",
                              "LTG_seizure_free",
                              "PHT_non_seizure_free", "CBZ_non_seizure_free",
                              "LTG_non_seizure_free",
                              "SCB seizure free", "SCB non seizure free"]
    filter_for_age_analyses = ["age known"]
    scb_names = ["PHT", "CBZ", "OXC", "LCM", "LTG", "ZON"]
    non_scb_aed_names = ["TPM", "VPA", "LEV", "ACTH", "LD", "MDL", "GBP", "KD", "PB", "VGB",
                         "CLB", "RUF", "STM", "PGN", "HC"]
    all_aed_names = scb_names + non_scb_aed_names
    effect_names = ["seizure_free", "seizure_reduction", "no_effect",
                    "worsening"]

    seizure_free_SCB = [["PHT_seizure_free", "CBZ_seizure_free",
                         "LTG_seizure_free", "OXC_seizure_free",
                         "LCM_seizure_free", "ZON_seizure_free"], "seizure free"]
    tmp_list = []
    seizure_free_AED = []
    seizure_reduction_AED = []
    no_effect_AED = []
    worsening_AED = []
    for aed_name in all_aed_names:
        seizure_free_AED.append(aed_name + "_seizure_free")
        seizure_reduction_AED.append(aed_name + "_seizure_reduction")
        no_effect_AED.append(aed_name + "_no_effect")
        worsening_AED.append(aed_name + "_worsening")
    seizure_free_AED = [seizure_free_AED, "seizure free"]
    seizure_reduction_AED = [seizure_reduction_AED, "seizure reduction"]
    no_effect_AED = [no_effect_AED, "no effect"]
    worsening_AED = [worsening_AED, "worsening"]

    seizure_reduction_SCB = [["PHT_seizure_reduction", "CBZ_seizure_reduction",
                              "LTG_seizure_reduction", "OXC_seizure_reduction",
                              "LCM_seizure_reduction", "ZON_seizure_reduction"], "seizure reduction"]
    no_effect_SCB = [["PHT_no_effect", "CBZ_no_effect",
                      "LTG_no_effect", "OXC_no_effect",
                      "LCM_no_effect", "ZON_no_effect"], "no effect"]
    worsening_SCB = [["PHT_worsening", "CBZ_worsening",
                      "LTG_worsening", "OXC_worsening",
                      "LCM_worsening", "ZON_worsening"], "worsening"]
    non_seizure_free_SCB = [["PHT_non_seizure_free", "CBZ_non_seizure_free",
                             "LTG_non_eizure_free", "OXC_non_seizure_free",
                             "LCM_non_seizure_free", "ZON_non_seizure_free"], "non seizure free"]

    # ____________________---------------_____________________
    do_region_analyses = False
    group_by_region = False
    stacked_version = False
    # if True, mean each plot of bar with variant freq, will concern one AED
    reverse_aed_region_dict = False

    holland_version = True
    only_msc = True
    # to put to False to draw Canal figure
    only_missence = True
    only_epilepsy = True

    do_age_analyses = False

    max_variant_freq = 0
    # the key will be a string representing a filter_option
    # the value will be a a dict with keys as instance of CanalRegion and the value is a list with number 1 is
    #  float representing
    # the relative frequency of variant.
    results_by_region_and_filter = dict()
    results_by_region_and_filter_stacked = dict()
    # --------------------_______________---------------------

    # filter_options = ["onset before 3 months", "onset after 3 months", "onset before 4 months", "onset after 4 months",
    #                   None]
    # filter_options = ["epilepsy", "epilepsy_missense", "no seizure"]
    # filter_options = [None]

    if do_region_analyses:
        if group_by_region and stacked_version:
            go_for_SCB_only = True
            if go_for_SCB_only:
                filter_options = [seizure_free_SCB[0], seizure_reduction_SCB[0], no_effect_SCB[0], worsening_SCB[0]]
                stacked_legend = [seizure_free_SCB[1], seizure_reduction_SCB[1], no_effect_SCB[1], worsening_SCB[1]]
            else:
                filter_options = [seizure_free_AED[0], seizure_reduction_AED[0], no_effect_AED[0], worsening_AED[0]]
                stacked_legend = [seizure_free_AED[1], seizure_reduction_AED[1], no_effect_AED[1], worsening_AED[1]]
        else:
            if group_by_region:

                list_to_use = seizure_free_SCB
                filter_options = list_to_use[0]

                # option for bar_charts grouped by region names
                title_for_group_by_region = None
                x_label_for_group_by_region = list_to_use[1]
                x_ticks_labels_dict = dict()
                for scb in list_to_use[0]:
                    aed_name = get_aed_name(scb)
                    x_ticks_labels_dict[scb] = aed_name
            else:
                filter_options = [None]  # filters_for_paper_fig1
                #
        x_ticks_vertical_rotation = False

    # filter_options = filters_for_paper_fig1_suppl
    # filter_options = filter_for_age_analyses
    # filter_options = [None]
    # filter_options = filters_all
    # filter_options = ["spasms"]
    filter_options = filters_for_mri

    # ___________ ----- A first loop to get the max frequency for each variant
    for filter_option in filter_options:
        if (not do_region_analyses) or stacked_version:
            break
        patient_mutation_dict = dict()
        # if True, we add the NaV1.6 patients whose filtered information is not known
        # we want it to be False for region analyses
        with_NA_filtered_patients = False

        for n, patient_series in scn8a_df.iterrows():
            patient = SCN8APatient(patient_series)
            if not keep_patient(patient=patient, filter_option=filter_option, only_msc=only_msc,
                                only_missence=only_missence, only_epilepsy=only_epilepsy):
                if (not with_NA_filtered_patients) or (not patient.NA_filtered_data):
                    continue
            # if n==0:
            #     print(f"{patient_series}")
            if patient.aa_change_position not in patient_mutation_dict:
                patient_mutation_dict[patient.aa_change_position] = []
            patient_mutation_dict[patient.aa_change_position].append(patient)

        fig = plt.figure(figsize=(12, 6))
        nav16 = Nav16(fig=fig)
        nav16.add_mutations(patient_mutation_dict)

        nav16.build_regions(holland_version=holland_version)
        max_variant_freq = max(nav16.get_max_variant_frequency(), max_variant_freq)
        plt.close()

    # print(f'max_variant_freq {max_variant_freq}')
    for filter_option_bis in filter_options:
        if (not stacked_version) or (not do_region_analyses):
            # trick in case we are not in the stacked version
            filter_option_bis = [filter_option_bis]
        for filter_option in filter_option_bis:
            # keep the SCN8APatient instance. The key is an int reprenseting the aa position and the value
            # is a list of SCN8APatient
            patient_mutation_dict = dict()
            # if True, we add the NaV1.6 patients whose filtered information is not known
            # we want it to be False for region analyses
            with_NA_filtered_patients = False

            for n, patient_series in scn8a_df.iterrows():
                patient = SCN8APatient(patient_series)
                if not keep_patient(patient=patient, filter_option=filter_option, only_msc=only_msc,
                                    only_missence=only_missence, only_epilepsy=only_epilepsy):
                    if (not with_NA_filtered_patients) or (not patient.NA_filtered_data):
                        continue

                # print(f"patient cohort {patient.cohort}, name {patient.patient_id}, mut_class {patient.mut_function}")
                if patient.aa_change_position not in patient_mutation_dict:
                    patient_mutation_dict[patient.aa_change_position] = []
                patient_mutation_dict[patient.aa_change_position].append(patient)

            fig = plt.figure(figsize=(12, 6))
            nav16 = Nav16(fig=fig)
            nav16.add_mutations(patient_mutation_dict)

            regions = nav16.build_regions(holland_version=holland_version)
            if do_region_analyses:
                nb_unique_aa_mutated = nav16.nb_unique_aa_mutated()
                nb_mutations = nav16.nb_mutations()
                print_misc_stat_from_holland_paper = False
                if print_misc_stat_from_holland_paper and (filter_option == "epilepsy_missense"):
                    print(f"{nb_unique_aa_mutated} of the {nb_mutations} "
                          f"{filter_option}-associated SCN8A variants occurred "
                          f"at unique amino acid "
                          f"locations within NaV1.6")
                    # loops_elements = [self.large_loop_d1_d2, self.large_loop_d2_d3, self.inactivation_gate_loop]
                    print(f"large_loop_d1_d2, relative frequency of variants: "
                          f"{np.round(nav16.large_loop_d1_d2.get_variants_freq() * 100, 2)}")
                    print(f"large_loop_d2_d3, relative frequency of variants: "
                          f"{np.round(nav16.large_loop_d2_d3.get_variants_freq() * 100, 2)}")
                    print(f"inactivation_gate_loop, relative frequency of variants: "
                          f"{np.round(nav16.inactivation_gate_loop.get_variants_freq() * 100, 2)}")
                    contingency_table = np.zeros((2, 2), dtype="int16")
                    # columns will be the gates, and the lines: relative nb of aa mutatated, relative nb of aa non mutated
                    # first version
                    contingency_table[0, 0] = int(nav16.inactivation_gate_loop.get_variants_freq() * 10000)
                    mean_two_other_loops = (nav16.large_loop_d1_d2.get_variants_freq() +
                                            nav16.large_loop_d2_d3.get_variants_freq()) / 2
                    contingency_table[0, 1] = int(mean_two_other_loops * 10000)
                    contingency_table[1, 0] = int((1 - nav16.inactivation_gate_loop.get_variants_freq()) * 10000)
                    contingency_table[1, 1] = int((1 - mean_two_other_loops) * 10000)

                    # second version
                    contingency_table[0, 0] = nav16.inactivation_gate_loop.nb_unique_aa_mutated()
                    sum_two_other_loops = nav16.large_loop_d1_d2.nb_unique_aa_mutated() + \
                                          nav16.large_loop_d2_d3.nb_unique_aa_mutated()
                    contingency_table[0, 1] = sum_two_other_loops
                    contingency_table[1, 0] = nav16.inactivation_gate_loop.nb_aa - \
                                              nav16.inactivation_gate_loop.nb_unique_aa_mutated()
                    sum_two_other_loops = (nav16.large_loop_d1_d2.nb_aa + nav16.large_loop_d2_d3.nb_aa) - \
                                          sum_two_other_loops
                    contingency_table[1, 1] = sum_two_other_loops
                    print(f"contingency_table {contingency_table}, mean_two_other_loops "
                          f"{mean_two_other_loops}")
                    oddsratio, p_value = stats.fisher_exact(contingency_table)
                    print(f"The inactivation gate has significantly more epilepsy-associated variants "
                          f"than the other intracellular loops "
                          f"for SCN8A (p={p_value})")

                    # unique mut
                    total_mut_in_pores = 0
                    total_mut_s5_6 = 0
                    total_mut_s4_5 = 0
                    all_pores_elements = []
                    for domain in nav16.domains.values():
                        for seg in np.arange(1, 6):
                            pore = domain.segments[seg].next_element
                            # total_mut_in_pores += pore.nb_unique_aa_mutated()
                            total_mut_in_pores += pore.nb_mutations()
                            if seg == 4:
                                # total_mut_s4_5 += pore.nb_unique_aa_mutated()
                                total_mut_s4_5 += pore.nb_mutations()
                            if seg == 5:
                                # total_mut_s5_6 += pore.nb_unique_aa_mutated()
                                total_mut_s5_6 += pore.nb_mutations()
                            all_pores_elements.append(pore)
                    all_pores_region = CanalRegion(region_id="Pore", canal_elements=all_pores_elements,
                                                   color=EpilepsiaColor.colors["bleu mauve"])
                    print(f"total_mut_in_pores {total_mut_in_pores}, "
                          f"total_mut_s5_6 {total_mut_s5_6} ({np.round((total_mut_s5_6/total_mut_in_pores)*100, 2)}%), "
                          f"total_mut_s4_5 {total_mut_s4_5} ({np.round((total_mut_s4_5/total_mut_in_pores)*100, 2)}%)")

                if group_by_region:
                    for region in regions:
                        if region not in results_by_region_and_filter:
                            results_by_region_and_filter[region] = dict()
                        if region not in results_by_region_and_filter_stacked:
                            results_by_region_and_filter_stacked[region.region_id] = dict()
                        if nav16.get_variants_freq() == 0:
                            relative_variant_freq = region.variants_frequency
                        else:
                            relative_variant_freq = region.variants_frequency / nav16.get_variants_freq()

                        results_by_region_and_filter[region][filter_option] = [relative_variant_freq,
                                                                               region.nb_aa_with_mutation,
                                                                               region.nb_mutated_patients,
                                                                               region.patients]
                        aed_name = get_aed_name(filter_option)
                        if aed_name not in results_by_region_and_filter_stacked[region.region_id]:
                            results_by_region_and_filter_stacked[region.region_id][aed_name] = []
                        results_by_region_and_filter_stacked[region.region_id][aed_name].append(
                            [relative_variant_freq,
                             region.nb_aa_with_mutation,
                             region.nb_mutated_patients,
                             region.patients])
                else:
                    nav16.plot_bar_variants_frequency(filter_option=filter_option, holland_version=holland_version,
                                                      path_results=path_results, max_variant_freq=max_variant_freq)
                plt.close()
            else:
                # print(nav16.get_str_structure())
                # print(f"Is it full?: {nav16.is_it_full()}")
                if do_age_analyses:
                    nav16_surrogates = build_n_surrogates(nav16=nav16, n_surrogate=1000, first_method=True)
                    cdf_age_a_plot_by_region(nav16=nav16, surrogates_nav16=nav16_surrogates, path_results=path_results)

                    # n_bins = np.arange(4, 20)
                    # for n_bins in n_bins:
                    #     region_distribution_by_age_interval(nav16=nav16, path_results=path_results, n_bins=n_bins)

                    n_bins = 9
                    title = f"distribution_region_for_age_intervals_{n_bins}_bins_real"
                    region_distribution_by_age_interval(nav16=nav16, path_results=path_results, n_bins=n_bins,
                                                        title=title)
                    for n_s, nav16_surrogate in enumerate(nav16_surrogates):
                        title = f"distribution_region_for_age_intervals_{n_bins}_bins_surrogate_{n_s}"
                        region_distribution_by_age_interval(nav16=nav16_surrogate, path_results=path_results,
                                                            n_bins=n_bins, title=title)

                    cdf_age_region(nav16=nav16, path_results=path_results)

                    nav16_boxplots(nav16, path_results=path_results)
                else:
                    nav16.print_mutations()
                    nav16.plot_element(filter_option=filter_option, path_results=path_results)
                    plt.close()
    if do_region_analyses and group_by_region:
        if stacked_version:
            if reverse_aed_region_dict:
                dict_to_use = reverse_dict_region_aed(results_by_region_and_filter_stacked)
                min_number_mut = -1
            else:
                dict_to_use = results_by_region_and_filter_stacked
                min_number_mut = 3
            if holland_version:
                x_ticks_label_size = 20
            else:
                x_ticks_label_size = 14
            plot_bar_variants_frequency_grouped_by_region_stacked(dict_to_use, path_results,
                                                                  holland_version=holland_version,
                                                                  max_variant_freq=-1,  # 8.5  # 20 zuberi
                                                                  title=None,
                                                                  effects_title=stacked_legend,
                                                                  x_label=None,
                                                                  x_ticks_vertical_rotation=x_ticks_vertical_rotation,
                                                                  min_number_mut=min_number_mut,
                                                                  x_ticks_label_size=x_ticks_label_size,
                                                                  forget_about_frequency=True)
        else:
            plot_bar_variants_frequency_grouped_by_region(results_by_region_and_filter, path_results,
                                                          holland_version=holland_version,
                                                          max_variant_freq=20,  # 8.5  # 20 zuberi
                                                          title=title_for_group_by_region,
                                                          x_label=x_label_for_group_by_region,
                                                          x_ticks_labels_dict=x_ticks_labels_dict,
                                                          x_ticks_vertical_rotation=x_ticks_vertical_rotation)

def reverse_dict_region_aed(results_by_region_and_filter_stacked):
    new_dict = dict()
    for region, aed_names in results_by_region_and_filter_stacked.items():
        for aed_name, lists_data in aed_names.items():
            if aed_name not in new_dict:
                new_dict[aed_name] = dict()
            new_dict[aed_name][region] = lists_data

    return new_dict


def build_surrogate(nav16, patients_list, holland_version=True, first_method=True):
    """
    two methods to build the surrogate:
    - First: we randomly give to all patients a pos among the first and last pos of the protein, making sure
    the number of patients by region is the same as the original nav16
    - Second: we randomly give to all patients a pos among the one on which patient were already.
    :param nav16:
    :param patients_list:
    :param holland_version:
    :param first_method:
    :return:
    """

    if not first_method:
        # first listing the aa_pos on which the patients have mutations
        list_of_available_aa_pos = [patient.aa_change_position for patient in patients_list]

    # first we copy the patients and then shuffle the patient list
    tmp_patients_list = []
    for patient in patients_list:
        tmp_patients_list.append(patient.get_a_copy())
    patients_list = tmp_patients_list
    random.shuffle(patients_list)
    if first_method:
        # surrogate will use nav16 and randomly distribute the patients, but keeping the same number of patients by
        # regions
        nb_patients_by_region = dict()
        nb_patients_added_by_region = dict()
        for region in nav16.regions_list:
            nb_patients_by_region[region.region_id] = region.nb_mutated_patients

        # taking a random value of aa_pos from aa_pos of regions that don't have the right numbers of patients yet
        # when the region is full, we remove its aa_pos of the pool of aa_pos avalable
        list_of_available_aa_pos = np.arange(nav16.first_position, nav16.last_position + 1)

        for patient in patients_list:
            aa_pos = random.choice(list_of_available_aa_pos)
            patient.aa_change_position = aa_pos
            # getting the region that just get a news patient
            region = nav16.get_element_at_position(aa_pos).region
            region_id = region.region_id
            nb_patients_added_by_region[region_id] = nb_patients_added_by_region.get(region_id, 0) + 1

            # then checking if the region is full
            if nb_patients_added_by_region[region_id] == nb_patients_by_region[region_id]:
                # then removing aa that are part of the region
                for canal_element in region.canal_elements:
                    canal_aa_pos = np.arange(canal_element.first_position, canal_element.last_position + 1)
                    list_of_available_aa_pos = np.setdiff1d(list_of_available_aa_pos, canal_aa_pos)
    else:
        for patient in patients_list:
            aa_pos_index = random.randrange(len(list_of_available_aa_pos))
            patient.aa_change_position = list_of_available_aa_pos[aa_pos_index]
            list_of_available_aa_pos.pop(aa_pos_index)

    patient_mutation_dict = dict()
    for patient in patients_list:
        if patient.aa_change_position not in patient_mutation_dict:
            patient_mutation_dict[patient.aa_change_position] = []
        patient_mutation_dict[patient.aa_change_position].append(patient)

    # first making a copy of nav16
    fig = plt.figure(figsize=(12, 6))
    surrogate_nav16 = Nav16(fig=fig)
    surrogate_nav16.add_mutations(patient_mutation_dict)

    surrogate_nav16.build_regions(holland_version=holland_version)

    # print("")
    # print("#"*70)
    # print("")
    # print("Surrogate data: ")
    # for region in surrogate_nav16.regions_list:
    #     print(f"## Region id {region.region_id}, nb {region.nb_mutated_patients}")
    #     for patient in region.patients:
    #         print(f"patient {patient.cohort} {patient.patient_id}")
    # print("")
    # print("//////"*10)
    # print("")

    plt.close()

    return surrogate_nav16


def build_n_surrogates(nav16, n_surrogate, first_method=True):
    # print("Real data: ")
    # for region in nav16.regions_list:
    #     print(f"## Region id {region.region_id}, nb {region.nb_mutated_patients}")
    #     for patient in region.patients:
    #         print(f"patient {patient.cohort} {patient.patient_id}")

    surrogates = []

    # getting the list of patients
    patients_list = []
    for patients in nav16.mutations.values():
        for patient in patients:
            patients_list.append(patient)

    for i in np.arange(n_surrogate):
        surrogates.append(build_surrogate(nav16=nav16, patients_list=patients_list, first_method=first_method))

    return surrogates


def region_distribution_by_age_interval(nav16, path_results, n_bins=15, title=None, show_fig=False):
    # looping on every patients, first determining min_age and max_age
    print(f"n_bins {n_bins}")
    min_age = 0
    max_age = 0
    n_patients = 0
    patients_sorted_by_ages = []
    for patients in nav16.mutations.values():
        for patient in patients:
            n_patients += 1
            patients_sorted_by_ages.append(patient)
            max_age = np.max((max_age, patient.age_onset))
    print(f"n_patients {n_patients}")
    patients_sorted_by_ages = sorted(patients_sorted_by_ages, key=lambda patient: patient.age_onset)
    # print(f"Sorted ages {[patient.age_onset for patient in patients_sorted_by_ages]}")
    # same number of patients by bin, it will determine the age interval.
    # the extra patients (the modulo), will go to the last bin
    n_patients_by_bin = n_patients // n_bins
    # then looping again to build the cdf
    # first getting the number of patient by region and each interval of age
    # each interval is 30 days, one month
    # we put +1 to be sure no patient is in the last interval, for bar display purposes
    go_for_old_version = False
    if go_for_old_version:
        age_intervals = np.geomspace(1, max_age + 1, n_bins)
    else:
        age_intervals = [0]
        age_interval_for_labels = [0]
    # print(f"age_intervals {age_intervals} max_age {max_age}")
    # another loop to determine age_intervals
    # patient_count = 0
    # for patients in nav16.mutations.values():
    #     for patient in patients:
    #         if patient_count == n_patients_by_bin:
    #             patient_count = 0

    regions_list = nav16.regions_list
    match_region_id_to_index = dict()
    region_colors = []
    for i, region in enumerate(regions_list):
        match_region_id_to_index[region.region_id] = i
        region_colors.append(region.color)
    n_regions = len(regions_list)

    # age_intervals = np.asarray(age_intervals)
    # n_intervals = len(age_intervals)

    fig = plt.figure(figsize=(10, 10), num=f"Regions distribution  over age intervals")
    axe_plot = fig.subplots()

    if go_for_old_version:
        region_distribution = np.zeros((n_regions, n_bins - 1))
        nb_mutated_patients = np.zeros((n_regions, n_bins - 1), dtype="int16")
    else:
        region_distribution = np.zeros((n_regions, n_bins))
        nb_mutated_patients = np.zeros((n_regions, n_bins), dtype="int16")

    if go_for_old_version:
        for patients in nav16.mutations.values():
            for patient in patients:
                patient_region_id = patient.region
                region_index = match_region_id_to_index[patient_region_id]
                patient_age = patient.age_onset
                if patient_age == 0:
                    patient_age = 1
                # determining interval index
                interval_index = bisect_right(age_intervals, patient_age) - 1
                # if interval_index == -1:
                #     print(f"patient_age {patient_age}")
                region_distribution[region_index, interval_index] += 1
    else:
        bin_index = 0
        patient_count = 0
        age_onset = 0
        next_patient = None
        # last_age_interval is
        # last_age_interval = 0
        for patient_index, patient in enumerate(patients_sorted_by_ages):
            patient_region_id = patient.region
            age_onset = patient.age_onset
            region_index = match_region_id_to_index[patient_region_id]
            region_distribution[region_index, bin_index] += 1
            patient_count += 1
            if (patient_count >= n_patients_by_bin) and (bin_index < (n_bins - 1)):
                # print(f"patient_count {patient_count}, bin_index {bin_index} age_onset {age_onset}")
                # if condition is used to make sure the upper bondary of a bin if not the same as the next bin
                next_patient = None
                if (patient_index + 1) < n_patients:
                    next_patient = patients_sorted_by_ages[patient_index + 1]
                if (next_patient is None) or (next_patient.age_onset != age_onset):
                    # print(f"added age_onset {age_onset}")
                    age_intervals.append(age_onset)
                    age_interval_for_labels.append(age_onset)
                    if next_patient is not None:
                        age_interval_for_labels.append(next_patient.age_onset)
                    else:
                        pass
                        # print("patient is None")
                    patient_count = 0
                    bin_index += 1
        # we look if last bin as less than 20% of the expected number of patient by bins and if so we add it to the
        # previous one
        last_bin_n_patients = np.sum(region_distribution[:, bin_index])
        if last_bin_n_patients < (n_patients_by_bin * 0.2):
            for region_index in np.arange(len(region_distribution)):
                region_distribution[region_index, bin_index - 1] += region_distribution[region_index, bin_index]
            if next_patient is not None:
                age_interval_for_labels = age_interval_for_labels[:-1]
            age_intervals[-1] = age_onset
            age_interval_for_labels[-1] = age_onset
        else:
            if next_patient is not None:
                age_intervals.append(age_onset)
                age_interval_for_labels.append(age_onset)
        age_intervals = np.asarray(age_intervals)
        # print(f"len(age_intervals) {len(age_intervals)}, age_intervals {age_intervals}, "
        #       f"n_patients_by_bin {n_patients_by_bin}")
        # print(f"age_interval_for_labels len {len(age_interval_for_labels)}:  {age_interval_for_labels}")
        n_bins = len(age_intervals)
        region_distribution = region_distribution[:, :(n_bins - 1)]

    nb_mutated_patients = np.copy(region_distribution).astype("int16")

    # putting value as percentage over a age interval
    for i in np.arange(n_bins - 1):
        if np.sum(region_distribution[:, i]) > 0:
            region_distribution[:, i] = (region_distribution[:, i] / np.sum(region_distribution[:, i])) * 100

    x_pos = np.arange(0, n_bins - 1)
    # print(f"x_pos {len(x_pos)}, region_distribution {len(region_distribution[0, :])}")
    plt_bar_list = []
    for i in np.arange(n_regions):
        linewidth = np.repeat(1, n_bins - 1)
        # linewidth[reg_variants_freq[i, :] == 0] = 0
        if i == 0:
            p = plt.bar(x_pos,
                        region_distribution[i, :], color=region_colors[i],
                        edgecolor=["black"] * (n_bins - 1), linewidth=linewidth)
        else:
            p = plt.bar(x_pos,
                        region_distribution[i, :], color=region_colors[i],
                        edgecolor=["black"] * (n_bins - 1), linewidth=linewidth,
                        bottom=np.sum(region_distribution[:i, :], axis=0))
        plt_bar_list.append(p)
        for index_x in np.arange(n_bins - 1):
            if region_distribution[i, index_x] > 0:
                y = 0
                if i > 0:
                    for before in np.arange(i):
                        y += region_distribution[before, index_x]
                y += region_distribution[i, index_x] / 2
                # if i == 0:
                #     color = "white"
                # else:
                color = "black"
                # fontsize = 6
                # if holland_version:
                fontsize = 9
                if len(x_pos) > 6:
                    fontsize -= 2
                    if len(x_pos) > 12:
                        fontsize -= 2
                    # if nb_aa_with_mutation[i, index_x] > 9:
                    #     fontsize -= 2

                plt.text(x=x_pos[index_x], y=y,
                         s=f"{np.round(region_distribution[i, index_x], 1)}% ({nb_mutated_patients[i, index_x]})",
                         color=color, zorder=22,
                         ha='center', va="center", fontsize=fontsize, fontweight='bold')

    x_ticks_label_size = 7
    if len(x_pos) > 10:
        x_ticks_label_size = 6
    # if len(age_intervals) > 20:
    #     x_ticks_label_size = 5
    x_ticks_vertical_rotation = len(x_pos) > 10
    x_ticks_pos = np.arange(n_bins - 1)
    if go_for_old_version:
        x_ticks_labels = [f"{np.round(age_intervals[i], 1)} - {np.round(age_intervals[i+1], 1)}"
                          for i in np.arange(n_bins - 1)]

    else:
        x_ticks_labels = [f"{age_interval_for_labels[i]} to {age_interval_for_labels[i+1]}"
                          for i in np.arange(0, len(age_interval_for_labels), 2)]

    if x_ticks_vertical_rotation:
        plt.xticks(x_ticks_pos, x_ticks_labels, fontsize=x_ticks_label_size,
                   fontweight="bold", rotation='vertical')
    else:
        plt.xticks(x_ticks_pos, x_ticks_labels, fontsize=x_ticks_label_size, fontweight="bold")

    axe_plot.spines['bottom'].set_visible(False)
    axe_plot.spines['right'].set_visible(False)
    axe_plot.spines['top'].set_visible(False)
    axe_plot.spines['left'].set_visible(False)
    axe_plot.set_ylim(-0.05, 101)

    y_pos = np.arange(0, 101, 10)
    # np.concatenante(y_pos, np.array([101])
    plt.yticks(y_pos, y_pos)
    # self.axe_plot.set_xlim(-1, self.total_width)

    plt.xlabel("Onset age intervals (days)", fontweight="bold", fontsize=12, labelpad=20)

    plt.ylabel('Proportion (%)', fontweight="bold", fontsize=14, labelpad=20)

    # grid
    # for i in np.arange(1, 101):
    #     plt.hlines(i, -0.4, n_intervals - 2 + 0.4, color="grey", linewidth=1, linestyles="dashed", zorder=1, alpha=0.2)

    # plt.title(f"{region}", fontweight="bold", fontsize=20)

    # rcParams['axes.titlepad'] = 2

    # new_effects_title = effects_title[:]
    # plt.legend()
    # plt.legend(plt_bar_list[::-1], [region.region_id for region in regions_list[::-1]],
    #            bbox_to_anchor=(0.98, 1), loc="upper left")
    plt.legend(plt_bar_list[::-1], [region.region_id for region in regions_list[::-1]],
               bbox_to_anchor=(-0.17, -0.15), loc="lower left", frameon=False)
    # "lower center" (0.5, 0.98)
    # path_results = "/Users/pappyhammer/Documents/academique/SCN8A/python_results"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    if title is None:
        fig.savefig(f'{path_results}/distribution_region_for_age_intervals_bins{n_bins}_'
                    f'{time_str}.pdf',
                    format="pdf")
    else:
        fig.savefig(f'{path_results}/{title}_{time_str}.pdf', format="pdf")
    if show_fig:
        plt.show()
    plt.close()


def cdf_age_a_plot_by_region(nav16, surrogates_nav16, path_results, title_fig=None, show_fig=False,
                             with_chi_square_stat=False):
    n_surrogates = len(surrogates_nav16)
    max_age = 0
    n_patients = 0

    for patients in nav16.mutations.values():
        for patient in patients:
            n_patients += 1
            max_age = np.max((max_age, patient.age_onset))

    # then looping again to build the cdf
    # first getting the number of patient by region and each interval of age
    # each interval is 5 days
    # if we want to do the chi_square to compare the distribution, then
    # we use the intervals used for Fig2B, such that the number of patients
    # by interval is similar, because there shouldn't be interval with zero patient
    # for  chi-square.
    if with_chi_square_stat:
        age_intervals = np.array([0, 10, 43, 105, 135, 210, 300])
    else:
        age_intervals = np.arange(0, max_age, 5)
    n_intervals = len(age_intervals)

    regions_list = nav16.regions_list
    match_region_id_to_index = dict()
    for i, region in enumerate(regions_list):
        match_region_id_to_index[region.region_id] = i
    n_regions = len(regions_list)

    patient_count = np.zeros((n_regions, n_intervals), dtype="float")
    patient_count_surrogate = np.zeros((n_surrogates, n_regions, n_intervals), dtype="float")
    list_nav16 = [nav16] + surrogates_nav16
    for index_nav16, each_nav16 in enumerate(list_nav16):
        for patients in each_nav16.mutations.values():
            for patient in patients:
                patient_region_id = patient.region
                region_index = match_region_id_to_index[patient_region_id]
                patient_age = patient.age_onset
                # if patient_age > max_age:
                #     continue
                # determining interval index
                interval_index = bisect_right(age_intervals, patient_age) - 1

                # print(f"index_nav16 {index_nav16}, patient.age_onset {patient.age_onset}, i {interval_index}")
                # if interval_index == -1:
                #     print(f"patient_age {patient_age}")
                if index_nav16 == 0:
                    patient_count[region_index, interval_index] += 1
                else:
                    patient_count_surrogate[index_nav16 - 1, region_index, interval_index] += 1

    # keeping a copy for stat (ks test) later
    original_patient_count = np.copy(patient_count)

    # putting value as percentage over a region
    # and doing sumcum
    for i, region_count in enumerate(patient_count):
        if np.sum(patient_count[i, :]) > 0:
            # print(f"{regions_list[i].region_id} {patient_count[i, :]}")
            patient_count[i, :] = (patient_count[i, :] / np.sum(patient_count[i, :])) * 100
        # now sum should be equal to 100
        patient_count[i, :] = np.cumsum(patient_count[i, :])
        # now last value of each line should be equal to 100

    for index_surrogate in np.arange(n_surrogates):
        for i in np.arange(n_regions):
            if np.sum(patient_count_surrogate[index_surrogate, i, :]) > 0:
                # print(f"{regions_list[i].region_id} {patient_count[i, :]}")
                patient_count_surrogate[index_surrogate, i, :] = \
                    (patient_count_surrogate[index_surrogate, i, :] /
                     np.sum(patient_count_surrogate[index_surrogate, i, :])) * 100
                # now sum should be equal to 100
                patient_count_surrogate[index_surrogate, i, :] = np.cumsum(
                    patient_count_surrogate[index_surrogate, i, :])
            # now last value of each line should be equal to 100

    for region_index in np.arange(n_regions):
        region = regions_list[region_index]
        if region.region_id == "N":
            continue

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                figsize=(8, 8))

        # plt.title("Cumulative distribution")

        for n in np.arange(n_surrogates):
            region_count = patient_count_surrogate[n, region_index, :]
            ax1.plot(age_intervals, region_count, c="lightgray", linewidth=1)

        # plotting 5 and 95 percentile
        region_count_95 = np.percentile(patient_count_surrogate[:, region_index, :], 95, axis=0)
        ax1.plot(age_intervals, region_count_95, c="black", linestyle="--", linewidth=1.5)
        region_count_5 = np.percentile(patient_count_surrogate[:, region_index, :], 5, axis=0)
        ax1.plot(age_intervals, region_count_5, c="black", linestyle="--", linewidth=1.5)

        # plotting surrogate median
        region_count = np.median(patient_count_surrogate[:, region_index, :], axis=0)
        ax1.plot(age_intervals, region_count, c="black", linewidth=2)

        # finding the distribution from the median, aka how many patient by age_intervals
        surrogate_patient_count = np.copy(region_count)
        for i in np.arange(1, len(surrogate_patient_count)):
            surrogate_patient_count[i] = surrogate_patient_count[i] - np.sum(surrogate_patient_count[:i])
        # now patient_count represents the percentage for each bin
        # now we decide how many patients we want in the distribution
        n_patients_surrogate = np.sum(original_patient_count[region_index, :])

        for i in np.arange(len(surrogate_patient_count)):
            surrogate_patient_count[i] = (surrogate_patient_count[i] / 100) * n_patients_surrogate

        # to_compare = np.copy(original_patient_count[region_index, :])

        if with_chi_square_stat:
            chisq, chi_p_value = stats.chisquare(f_obs=original_patient_count[region_index, :],
                                                 f_exp=surrogate_patient_count)
            print(f"{region.region_id}: chisq: D {np.round(chisq, 5)}, chi_p_value {np.round(chi_p_value, 6)} ")

        region_count = patient_count[region_index, :]

        ax1.plot(age_intervals, region_count, c=region.color, linewidth=3, alpha=0.8)

        # plt.legend([region.region_id for region in regions_list],
        #            loc='lower right')
        plt.xlabel("Onset age (days)", fontweight="bold", fontsize=12, labelpad=10)
        ax1.set_xscale("log")
        plt.ylabel("Cumulative percent", fontweight="bold", fontsize=12, labelpad=10)
        plt.title(f"{region.region_id}", fontweight="bold", fontsize=20)
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        if not with_chi_square_stat:
            if title_fig is None:
                fig.savefig(f'{path_results}/cumulative_distribution_age_region_{region.region_id}_{time_str}.pdf',
                            format="pdf")
            else:
                fig.savefig(f'{path_results}/{title_fig}_{region.region_id}_{time_str}.pdf',
                            format="pdf")
            if show_fig:
                plt.show()
        plt.close()


# stats.ks_2samp(rvs1, rvs2)


def cdf_age_region(nav16, path_results, title_fig=None, show_fig=False):
    # looping on every patients, first determining min_age and max_age
    min_age = 0
    max_age = 0

    for patients in nav16.mutations.values():
        for patient in patients:
            max_age = np.max((max_age, patient.age_onset))

    # then looping again to build the cdf
    # first getting the number of patient by region and each interval of age
    # each interval is 30 days, one month
    age_intervals = np.arange(0, max_age, 5)
    n_intervals = len(age_intervals)
    regions_list = nav16.regions_list
    match_region_id_to_index = dict()
    for i, region in enumerate(regions_list):
        match_region_id_to_index[region.region_id] = i
    n_regions = len(regions_list)

    patient_count = np.zeros((n_regions, n_intervals), dtype="float")
    # print(f"age_intervals {age_intervals} len {len(age_intervals)}")
    for patients in nav16.mutations.values():
        for patient in patients:
            patient_region_id = patient.region
            region_index = match_region_id_to_index[patient_region_id]
            patient_age = patient.age_onset
            # determining interval index
            interval_index = bisect_right(age_intervals, patient_age) - 1
            # if interval_index == -1:
            #     print(f"patient_age {patient_age}")
            patient_count[region_index, interval_index] += 1

    # putting value as percentage over a region
    # and doing sumcum
    for i, region_count in enumerate(patient_count):
        if np.sum(patient_count[i, :]) > 0:
            # print(f"{regions_list[i].region_id} {patient_count[i, :]}")
            patient_count[i, :] = (patient_count[i, :] / np.sum(patient_count[i, :])) * 100
        # now sum should be equal to 100
        patient_count[i, :] = np.cumsum(patient_count[i, :])
        # now last value of each line should be equal to 100

    # now we plot the values

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            figsize=(8, 8))

    # plt.title("Cumulative distribution")

    for i, region_count in enumerate(patient_count):
        if regions_list[i].region_id == "N":
            continue
        ax1.plot(age_intervals, region_count, c=regions_list[i].color, linewidth=3)

    for i, region in enumerate(regions_list):
        if region.region_id == "N":
            regions_list.pop(i)
            break

    plt.legend([region.region_id for region in regions_list],
               loc='lower right')
    plt.xlabel("Onset age (days)", fontweight="bold", fontsize=12, labelpad=10)
    ax1.set_xscale("log")
    plt.ylabel("Cumulative percent", fontweight="bold", fontsize=12, labelpad=10)
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    if title_fig is None:
        fig.savefig(f'{path_results}/cumulative_distribution_age_vs_region_{time_str}.pdf',
                format="pdf")
    else:
        fig.savefig(f'{path_results}/{title_fig}_{time_str}.pdf',
                    format="pdf")
    if show_fig:
        plt.show()
    plt.close()


# do the boxplots
def nav16_boxplots(nav16, path_results, title_fig=None, show_fig=False):
    list_patients_grouped_by_unique_mutation = nav16.patients_grouped_by_unique_mutation()
    # print some stat about mutations
    print(f"Nb patients total: {nav16.nb_mutations()}")
    print(f"nb_unique_mutation: {len(list_patients_grouped_by_unique_mutation)}")
    # do a multiple box plot representing ages for a given mutation
    # list containing a list for each mutation
    age_data = []
    mutation_name = []
    box_colors = []
    nb_patients_by_mut = []
    i = 0
    # we keep only recurrent mutation
    for list_patients in list_patients_grouped_by_unique_mutation:
        if len(list_patients) > 1:
            age_data.append([])
            # print("")
            # print(f"### Group {i+1}: {list_patients[0].aa_change_one_letter} ###")
            for n_p, patient in enumerate(list_patients):
                # to make sure we have not 0 value
                patient_onset_age = patient.age_onset if patient.age_onset > 0 else 1
                age_data[i].append(np.log10(patient_onset_age))
                # print(f"{patient.cohort} {patient.patient_id}")
                if n_p == 0:
                    mutation_name.append(patient.aa_change_one_letter)
                    box_colors.append(patient.region.color)
                    nb_patients_by_mut.append(len(list_patients))
            i += 1

    print(f"\nnb_ recurrent mutation: {i}")

    n_mut = len(mutation_name)
    widths_box_plot = np.array(nb_patients_by_mut)
    widths_box_plot = (np.sqrt(widths_box_plot) - 0.8) * 0.3
    fig = plt.figure(figsize=(8, 8), num=f"Age vs mutation")
    axe_plot = fig.subplots()

    bp = axe_plot.boxplot(age_data, widths=widths_box_plot)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['medians'], color='black')

    # Now fill the boxes with desired colors
    # medians = list(range(n_mut))
    # for i in range(n_mut):
    #     box = bp['boxes'][i]
    #     boxX = []
    #     boxY = []
    #     for j in range(5):
    #         boxX.append(box.get_xdata()[j])
    #         boxY.append(box.get_ydata()[j])
    #     boxCoords = np.column_stack([boxX, boxY])
    #     boxPolygon = Polygon(boxCoords, facecolor=box_colors[i])
    #     axe_plot.add_patch(boxPolygon)

    # Now draw the median lines back over what we just filled in
    # med = bp['medians'][i]
    # medianX = []
    # medianY = []
    # for j in range(2):
    #     medianX.append(med.get_xdata()[j])
    #     medianY.append(med.get_ydata()[j])
    #     axe_plot.plot(medianX, medianY, 'k')
    #     medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    # axe_plot.plot([np.average(med.get_xdata())], [np.average(age_data[i])],
    #          color='w', marker='*', markeredgecolor='k')

    # axe_plot.spines['bottom'].set_visible(False)
    # axe_plot.spines['right'].set_visible(False)
    # axe_plot.spines['top'].set_visible(False)
    # axe_plot.spines['left'].set_visible(False)

    # axe_plot.set_ylim(-0.05, max_variant_freq)

    y_pos = np.arange(1, 11)
    y_pos = np.concatenate((y_pos, np.arange(20, 101, 10)))
    y_pos = np.concatenate((y_pos, np.arange(200, 1001, 100)))
    y_pos = np.concatenate((y_pos, np.array([2000, 3000])))

    y_pos_label_tmp = np.copy(y_pos)
    y_pos_label_tmp = list(y_pos_label_tmp)
    y_pos_label = y_pos_label_tmp[:]
    for i, pos in enumerate(y_pos_label_tmp):
        if pos in [1, 10, 100, 1000, 3000]:
            pass
        else:
            y_pos_label[i] = ""
    y_pos_label = [""] + y_pos_label

    plt.yticks(np.concatenate((np.array([0]), np.log10(y_pos))), y_pos_label)
    # self.axe_plot.set_xlim(-1, self.total_width)

    plt.xlabel(f"Mutation", fontweight="bold", fontsize=12, labelpad=10)
    axe_plot.set_xticklabels(mutation_name, rotation=90, fontsize=8)
    plt.ylabel('Age of onset (days)', fontweight="bold", fontsize=14, labelpad=10)
    # grid
    for i in np.log10([90, 120]):
        plt.hlines(i, 0, n_mut, color="grey", linewidth=1, linestyles="dashed", zorder=1, alpha=0.2)

    rcParams['axes.titlepad'] = 20

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    if title_fig is None:
        fig.savefig(f'{path_results}/age_vs_mutation_boxplots_'
                    f'{time_str}.pdf',
                    format="pdf")
    else:
        fig.savefig(f'{path_results}/{title_fig}_'
                    f'{time_str}.pdf',
                    format="pdf")
    if show_fig:
        plt.show()
    plt.close()


def plot_bar_variants_frequency_grouped_by_region_stacked(results_by_region_and_filter, path_results,
                                                          holland_version, max_variant_freq,
                                                          effects_title,
                                                          x_ticks_label_size,
                                                          x_label, title=None,
                                                          x_ticks_vertical_rotation=True,
                                                          min_number_mut=3,
                                                          forget_about_frequency=False,
                                                          title_fig=None):
    # if forget_about_frequency is True, means we stacked the numbers of patients
    debug_mode = False
    max_variant_freq = 0
    x_ticks_label_size_changed = False
    new_results_by_region_and_filter = dict()
    # list of set, each set contains the unique patients for each effect
    nb_unique_patients_by_effect = dict()
    for region in results_by_region_and_filter:
        nb_unique_patients_by_effect[region] = list()
        for effect in effects_title:
            nb_unique_patients_by_effect[region].append(set())
    # list of string, AED to remove because not enough patient
    aed_to_remove = []
    nb_mutations_dict = dict()
    # getting the max and filtering
    for region, filter_option_dict in results_by_region_and_filter.items():
        n_effects = 0
        # seizure free, reduction, no effect, worsening
        i = 0
        for filter_option, lists_data in filter_option_dict.items():
            if n_effects == 0:
                n_effects = len(lists_data)
            reg_variants_freq_sum = np.zeros(len(filter_option_dict))
            sum_patients = 0
            for n, list_data in enumerate(lists_data):
                reg_variants_freq_sum[i] += list_data[0]
                if filter_option not in nb_mutations_dict:
                    nb_mutations_dict[filter_option] = list_data[2]
                else:
                    nb_mutations_dict[filter_option] += list_data[2]
                sum_patients += list_data[2]
            i += 1
            if forget_about_frequency:
                if sum_patients > max_variant_freq:
                    max_variant_freq = sum_patients
            else:
                if np.sum(reg_variants_freq_sum) > max_variant_freq:
                    max_variant_freq = np.sum(reg_variants_freq_sum)
    if min_number_mut > 0:
        for region, filter_option_dict in results_by_region_and_filter.items():
            new_results_by_region_and_filter[region] = dict()
            for filter_option, n_mut in nb_mutations_dict.items():
                # filtering
                if n_mut >= min_number_mut:
                    new_results_by_region_and_filter[region][filter_option] = \
                        results_by_region_and_filter[region][filter_option]

        # filtering with min number of patients mutated
        results_by_region_and_filter = new_results_by_region_and_filter

    for region, filter_option_dict in results_by_region_and_filter.items():
        local_max_variant_freq = 0
        if debug_mode:
            print(f"region {region}")

        fig = plt.figure(figsize=(10, 10), num=f"{region}: relative frequency of variants for each filter")
        axe_plot = fig.subplots()

        n_aeds = len(filter_option_dict)
        x_ticks_labels = []
        reg_variants_freq = None
        nb_aa_with_mutation = None
        nb_mutated_patients = None
        i = 0
        n_effects = 0
        # seizure free, reduction, no effect, worsening
        bar_colors = [EpilepsiaColor.colors["bleu fonce"], EpilepsiaColor.colors["bleu normal"],
                      EpilepsiaColor.colors["bleu clair"], EpilepsiaColor.colors["rose fonce"]]
        for filter_option, lists_data in filter_option_dict.items():
            if n_effects == 0:
                n_effects = len(lists_data)
            if debug_mode:
                print(f"filter_option {filter_option}, n_effects {n_effects}")
            if reg_variants_freq is None:
                reg_variants_freq = np.zeros((n_effects, n_aeds))
                nb_aa_with_mutation = np.zeros((n_effects, n_aeds), dtype="int16")
                nb_mutated_patients = np.zeros((n_effects, n_aeds), dtype="int16")
                patients_by_categories = dict()
            # bar_colors.append(region.color)
            # first_value is the relative variant frequency for a given effect (seizure free, etc...)
            reg_variants_freq_sum = np.zeros(len(filter_option_dict))
            sum_patients = 0
            for n, list_data in enumerate(lists_data):
                reg_variants_freq_sum[i] += list_data[0]
                if forget_about_frequency:
                    reg_variants_freq[n, i] = list_data[2]
                    sum_patients += list_data[2]
                    # updating the set
                    nb_unique_patients_by_effect[region][n].update(list_data[3])
                else:
                    reg_variants_freq[n, i] = list_data[0]
                nb_aa_with_mutation[n, i] = list_data[1]
                nb_mutated_patients[n, i] = list_data[2]
                if n not in patients_by_categories:
                    patients_by_categories[n] = dict()
                patients_by_categories[n][i] = list_data[3]
            if forget_about_frequency:
                if sum_patients > local_max_variant_freq:
                    local_max_variant_freq = sum_patients
            else:
                if np.sum(reg_variants_freq_sum) > local_max_variant_freq:
                    local_max_variant_freq = np.sum(reg_variants_freq_sum)
            # print(f"Region {region.region_id}, relative frequency of variants: {np.round(reg_variants_freq[i], 2)}")
            x_ticks_labels.append(filter_option)
            i += 1

        x_pos = np.arange(0, n_aeds)
        plt_bar_list = []
        for i in np.arange(n_effects):
            linewidth = np.repeat(1, n_aeds)
            # linewidth[reg_variants_freq[i, :] == 0] = 0
            if i == 0:
                p = plt.bar(x_pos,
                            reg_variants_freq[i, :], color=bar_colors[i],
                            edgecolor=["black"] * n_aeds, linewidth=linewidth)
            else:
                p = plt.bar(x_pos,
                            reg_variants_freq[i, :], color=bar_colors[i],
                            edgecolor=["black"] * n_aeds, linewidth=linewidth,
                            bottom=np.sum(reg_variants_freq[:i, :], axis=0))
            plt_bar_list.append(p)
            for index_x in np.arange(n_aeds):
                if reg_variants_freq[i, index_x] > 0:
                    y = 0
                    if i > 0:
                        for before in np.arange(i):
                            y += reg_variants_freq[before, index_x]
                    y += reg_variants_freq[i, index_x] / 2
                    if i == 0:
                        color = "white"
                    else:
                        color = "black"
                    # fontsize = 6
                    # if holland_version:
                    fontsize = 14
                    if len(x_pos) > 10:
                        fontsize -= 3
                        # if nb_aa_with_mutation[i, index_x] > 9:
                        #     fontsize -= 2
                    display_mean_age = True
                    if display_mean_age:
                        patients = patients_by_categories[i][index_x]
                        mean_age = []
                        age_str = ""
                        for patient in patients:
                            if (patient.age_onset != "nan") and (patient.age_onset > 0):
                                mean_age.append(patient.age_onset)
                        if len(mean_age) > 0:
                            age_str += "\n\n"
                            for index_patient, age_patient in enumerate(mean_age):
                                age_str += f"{int(age_patient)}"
                                if index_patient < (len(mean_age) - 1):
                                    age_str += f"\n"
                            # mean_age_nb = np.mean(mean_age)
                            # age_str += f"{int(mean_age_nb)}"
                            # if len(mean_age) > 1:
                            #     std_age = np.std(mean_age)
                            #     age_str += f" +/- {int(std_age)}"

                        plt.text(x=x_pos[index_x], y=y,
                                 s=f"{nb_aa_with_mutation[i, index_x]} / {nb_mutated_patients[i, index_x]}{age_str}",
                                 color=color, zorder=22,
                                 ha='center', va="center", fontsize=fontsize, fontweight='bold')
                    else:
                        plt.text(x=x_pos[index_x], y=y,
                                 s=f"{nb_aa_with_mutation[i, index_x]} / {nb_mutated_patients[i, index_x]}",
                                 color=color, zorder=22,
                                 ha='center', va="center", fontsize=fontsize, fontweight='bold')

        properties = {'weight': 'bold'}

        SCB_to_color_dict = dict()
        SCB_to_color_dict["PHT"] = EpilepsiaColor.colors["bleu clair"]
        SCB_to_color_dict["CBZ"] = EpilepsiaColor.colors["orange clair"]
        SCB_to_color_dict["OXC"] = EpilepsiaColor.colors["vert fonce"]
        SCB_to_color_dict["LCM"] = EpilepsiaColor.colors["jaune"]
        SCB_to_color_dict["LTG"] = EpilepsiaColor.colors["rose fonce"]
        SCB_to_color_dict["ZON"] = EpilepsiaColor.colors["vert fonce"]

        if (len(x_pos) > 10) and (not x_ticks_label_size_changed):
            x_ticks_label_size -= 2
            x_ticks_label_size_changed = True
        if x_ticks_vertical_rotation:
            plt.xticks(x_pos, x_ticks_labels, fontsize=x_ticks_label_size, fontweight="bold", rotation='vertical')
        else:
            plt.xticks(x_pos, x_ticks_labels, fontsize=x_ticks_label_size, fontweight="bold")

        colors = []
        for label in x_ticks_labels:
            if label in SCB_to_color_dict:
                colors.append(SCB_to_color_dict[label])
            else:
                colors.append("black")
        for color, tick in zip(colors, axe_plot.xaxis.get_major_ticks()):
            tick.label1.set_color(color)

        axe_plot.spines['bottom'].set_visible(False)
        axe_plot.spines['right'].set_visible(False)
        axe_plot.spines['top'].set_visible(False)
        axe_plot.spines['left'].set_visible(False)
        axe_plot.set_ylim(-0.05, max_variant_freq)

        y_pos = np.arange(0, int(np.ceil(max_variant_freq)) + 1)
        plt.yticks(y_pos, y_pos)
        # self.axe_plot.set_xlim(-1, self.total_width)

        # plt.xlabel(f"{x_label}", fontweight="bold", fontsize=12, labelpad=20)
        if forget_about_frequency:
            plt.ylabel('Number of patients', fontweight="bold", fontsize=24, labelpad=20)
        else:
            plt.ylabel('Relative frequency of variants', fontweight="bold", fontsize=14, labelpad=20)
        # grid
        for i in np.arange(1, int(np.ceil(local_max_variant_freq) + 1)):
            plt.hlines(i, -0.4, n_aeds - 1 + 0.4, color="grey", linewidth=1, linestyles="dashed", zorder=1, alpha=0.2)
        # if title is not None:
        #     plt.title(f"{title}")
        # axe_plot.set_title(f"{region.region_id}", fontweight="bold", fontsize=14, pad=20)
        plt.title(f"{region}", fontweight="bold", fontsize=24)
        rcParams['axes.titlepad'] = 20
        new_effects_title = effects_title[:]
        if forget_about_frequency:
            for i, title in enumerate(effects_title):
                new_effects_title[i] = effects_title[i] + " (" + \
                                       str(len(nb_unique_patients_by_effect[region][i])) + ")"

        plt.legend(plt_bar_list[::-1], new_effects_title[::-1])

        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        if holland_version:
            paper_inspiration = "Holland"
        else:
            paper_inspiration = "Zuberi"
        if title_fig is None:
            fig.savefig(f'{path_results}/{region}_nav16_variant_frequency_stacked_{paper_inspiration}_'
                        f'{time_str}.pdf',
                        format="pdf")
        else:
            fig.savefig(f'{path_results}/{region}_{title_fig}_'
                        f'{time_str}.pdf',
                        format="pdf")
        # plt.show()
        plt.close()


def get_aed_name(data):
    # data in the form of "PTH_seizure_free"
    index_ = data.index('_')
    return data[:index_]


def plot_bar_variants_frequency_grouped_by_region(results_by_region_and_filter, path_results,
                                                  holland_version, max_variant_freq, x_ticks_labels_dict=None,
                                                  x_label=None, title=None,
                                                  x_ticks_vertical_rotation=True):
    for region, filter_option_dict in results_by_region_and_filter.items():
        fig = plt.figure(figsize=(8, 8), num=f"{region.region_id}: relative frequency of variants for each filter")
        axe_plot = fig.subplots()

        n_filters = len(filter_option_dict)
        reg_variants_freq = np.zeros(n_filters)
        x_ticks_labels = []
        bar_colors = [region.color] * n_filters
        nb_aa_with_mutation = np.zeros(n_filters, dtype="int16")
        nb_mutated_patients = np.zeros(n_filters, dtype="int16")
        i = 0
        for filter_option, list_data in filter_option_dict.items():
            # bar_colors.append(region.color)
            # first_value is the relative variant frequency
            reg_variants_freq[i] = list_data[0]
            nb_aa_with_mutation[i] = list_data[1]
            nb_mutated_patients[i] = list_data[2]
            # print(f"Region {region.region_id}, relative frequency of variants: {np.round(reg_variants_freq[i], 2)}")
            if x_ticks_labels_dict is not None:
                x_ticks_labels.append(x_ticks_labels_dict[filter_option])
            i += 1

        x_pos = np.arange(0, n_filters)
        linewidth = np.repeat(1, n_filters)
        linewidth[reg_variants_freq == 0] = 0
        plt.bar(x_pos,
                reg_variants_freq, color=bar_colors,
                edgecolor=["black"] * n_filters, linewidth=linewidth)
        for i in np.arange(n_filters):
            if reg_variants_freq[i] > 0:
                y = reg_variants_freq[i] / 2
                color = "grey"
                if holland_version:
                    color = "black"
                # fontsize = 6
                # if holland_version:
                fontsize = 10
                plt.text(x=x_pos[i], y=y,
                         s=f"{nb_aa_with_mutation[i]} / {nb_mutated_patients[i]}",  # \n({region.nb_total_aa})",
                         color=color, zorder=22,
                         ha='center', va="center", fontsize=fontsize, fontweight='bold')

        # self.nb_mutated_patients

        # show variant frequency for the whole protein
        # end_y = 13.4
        # if holland_version:
        #     end_y = 5.4
        # plt.hlines(self.get_variants_freq(), -0.4, end_y, color="grey", linewidth=1, linestyles="dashed")

        # print(f"NaV1.6, relative frequency of variants: {np.round(self.get_variants_freq(), 4)}")
        # print(f"NaV1.6 nb_unique aa: {self.nb_unique_aa_mutated()}")
        # print(f"NaV1.6 nb aa: {self.nb_aa}")
        properties = {'weight': 'bold'}

        if len(x_ticks_labels) > 0:
            if x_ticks_vertical_rotation:
                plt.xticks(x_pos, x_ticks_labels, fontweight="bold", rotation='vertical')
            else:
                plt.xticks(x_pos, x_ticks_labels, fontweight="bold")

        # self.axe_plot.axes.get_xaxis().set_visible(False)
        # self.axe_plot.axes.get_yaxis().set_visible(False)
        axe_plot.spines['bottom'].set_visible(False)
        axe_plot.spines['right'].set_visible(False)
        axe_plot.spines['top'].set_visible(False)
        axe_plot.spines['left'].set_visible(False)
        axe_plot.set_ylim(-0.05, max_variant_freq)

        y_pos = np.arange(0, int(np.ceil(max_variant_freq)))
        plt.yticks(y_pos, y_pos)
        # self.axe_plot.set_xlim(-1, self.total_width)

        if x_label is not None:
            plt.xlabel(f"{x_label}", fontweight="bold", fontsize=12, labelpad=20)
        plt.ylabel('Relative frequency of variants', fontweight="bold", fontsize=12, labelpad=20)
        # if title is not None:
        #     plt.title(f"{title}")
        plt.title(f"{region.region_id}", fontweight="bold")
        # path_results = "/Users/pappyhammer/Documents/academique/SCN8A/python_results"
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        if holland_version:
            paper_inspiration = "Holland"
        else:
            paper_inspiration = "Zuberi"
        fig.savefig(f'{path_results}/{region.region_id}_nav16_variant_frequency_{paper_inspiration}_'
                    f'{x_label}_{time_str}.pdf',
                    format="pdf")
        # plt.show()
        plt.close()


def keep_patient(patient, filter_option, only_msc=False, only_missence=False, only_epilepsy=True):
    if only_msc:
        if not patient.msc:
            return False
    if only_missence:
        if patient.mut_function != 0:
            return False

    if only_epilepsy:
        # keeping only patients with seizures
        if patient.seizure_type_onset == -3:
            return False

    if filter_option is None:
        return True

    # "onset before 3 months"
    if filter_option == "onset before 3 months":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset <= 90

    if filter_option == "onset after 3 months":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset > 90

    if filter_option == "onset before 1 month":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset <= 30

    if filter_option == "onset after 1 month":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset > 30

    if filter_option == "onset before 1 month with AED info":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return (patient.age_onset <= 30) and (patient.aed_efficacy != -1)

    if filter_option == "onset after 1 month with AED info":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return (patient.age_onset > 30) and (patient.aed_efficacy != -1)

    # "onset before 1 month", "onset after 1 month",

    if filter_option == "onset before 3 months with AED info":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return (patient.age_onset <= 90) and (patient.aed_efficacy != -1)

    if filter_option == "onset after 3 months with AED info":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return (patient.age_onset > 90) and (patient.aed_efficacy != -1)

    # "onset before 3 months with AED info", "onset after 3 months with AED info",

    if filter_option == "onset before 4 months":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset <= 120

    if filter_option == "age known":
        if patient.age_onset >= 0:
            return True
        return False

    if filter_option == "onset after 4 months":
        if patient.age_onset == -2:
            patient.NA_filtered_data = True
        if patient.age_onset == -1:
            return False
        return patient.age_onset > 120

    if filter_option == "no seizure":
        return patient.seizure_type_onset == -3
    if filter_option == "group 1":
        if patient.group_1_NA:
            patient.NA_filtered_data = True
        return patient.group_1
    if filter_option == "group 1a":
        if patient.group_1_NA:
            patient.NA_filtered_data = True
        return patient.group_1_a
    if filter_option == "group 1b":
        if patient.group_1_NA:
            patient.NA_filtered_data = True
        return patient.group_1_b
    if filter_option == "group 2":
        if patient.group_2_NA:
            patient.NA_filtered_data = True
        return patient.group_2

    if filter_option == "epilepsy":
        return patient.seizure_type_onset != -3
    if filter_option == "epilepsy_missense":
        return (patient.seizure_type_onset != -3) and (patient.mut_function == 0)
    if filter_option == "other type of seizure":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == -2
    if filter_option == "spasms":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 1
    if filter_option == "mj":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 0
    if filter_option == "tonic seizures":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 4
    if filter_option == "GTCS":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 2
    if filter_option == "convulsive status":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 3
    if filter_option == "focal seizure":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 5
    if filter_option == "Focal seizure -> generalized":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 6
    if filter_option == "febrile seizure":
        if patient.seizure_type_onset == -1:
            patient.NA_filtered_data = True
        return patient.seizure_type_onset == 7
    if filter_option == "normal development after onset":
        if patient.dev_after == -1:
            patient.NA_filtered_data = True
        return patient.dev_after == 0
    if filter_option == "delayed development after onset":
        if patient.dev_after == -1:
            patient.NA_filtered_data = True
        return patient.dev_after >= 1
    if filter_option == "delayed development without regression after onset":
        if patient.dev_after == -1:
            patient.NA_filtered_data = True
        return patient.dev_after == 1
    if filter_option == "delayed development with regression after onset":
        if patient.dev_after == -1:
            patient.NA_filtered_data = True
        return patient.dev_after == 2
    if filter_option == "normal development before onset":
        if patient.dev_before == -1:
            patient.NA_filtered_data = True
        return patient.dev_before == 0
    if filter_option == "delayed development before onset":
        if patient.dev_before == -1:
            patient.NA_filtered_data = True
        return patient.dev_before == 1
    if filter_option == "normal EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 0
    if filter_option == "first EEG known":
        return patient.eeg_onset != -1
    if filter_option == "Non paroxysmal abnormality EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 1
    if filter_option == "focal paroxysmal abnormality EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 2
    if filter_option == "multifocal paroxysmal abnormality EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 3
    if filter_option == "paroxysmal abnormality EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return (patient.eeg_onset == 3) or (patient.eeg_onset == 2) or (patient.eeg_onset == 4)
    if filter_option == "hypsarrythmia":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 4
    if filter_option == "no worries on EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return patient.eeg_onset == 5
    if filter_option == "non paroxysmal group EEG":
        if patient.eeg_onset == -1:
            patient.NA_filtered_data = True
        return (patient.eeg_onset == 5) or (patient.eeg_onset == 1) or (patient.eeg_onset == 0)
    if filter_option == "sudep":
        return patient.sudep >= 0
    if filter_option == "AED seizure control":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        return patient.aed_efficacy == 0
    if filter_option == "AED no seizure control":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        return patient.aed_efficacy == 1
    if filter_option == "AED partial seizure control":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        return patient.aed_efficacy == 2
    if filter_option == "SCB seizure free":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] == 1:
                return True
        return False

    if filter_option == "SCB non seizure free":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] > 1:
                return True
        return False

    if filter_option == "SCB worsening":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] == 4:
                return True
        return False

    if filter_option == "SCB seizure reduction":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] == 2:
                return True
        return False

    if filter_option == "SCB no effect":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] == 3:
                return True
        return False

    if filter_option == "SCB no effect or worsening":
        if patient.aed_efficacy == -1:
            patient.NA_filtered_data = True
        for scb in patient.scb_names:
            if patient.aed_efficacy_dict[scb] > 2:
                return True
        return False

    # "SCB seizure reduction", "SCB no effect",

    for aed in patient.aed_names:
        if filter_option == f"{aed}_seizure_free":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] == 1:
                return True
            return False
        if filter_option == f"{aed}_non_seizure_free":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] > 1:
                return True
            return False

        if filter_option == f"{aed}_no_effect_or_worsening":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] > 2:
                return True
            return False

        if filter_option == f"{aed}_no_effect":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] == 3:
                return True
            return False

        if filter_option == f"{aed}_worsening":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] == 4:
                return True
            return False

        if filter_option == f"{aed}_seizure_reduction":
            if patient.aed_efficacy == -1:
                patient.NA_filtered_data = True
            if patient.aed_efficacy_dict[aed] == 2:
                return True
            return False
        if filter_option == "ofc birth micro":
            if patient.ofc_birth == 1:
                return True
            return False
        if filter_option == "ofc birth macro":
            if patient.ofc_birth == 2:
                return True
            return False
        if filter_option == "ofc birth normal":
            if patient.ofc_birth == 0:
                return True
            return False

        if filter_option == "ofc evolution micro":
            if patient.ofc_evolution == 1:
                return True
            return False
        if filter_option == "ofc evolution macro":
            if patient.ofc_evolution == 2:
                return True
            return False
        if filter_option == "ofc evolution normal":
            if patient.ofc_evolution == 0:
                return True
            return False

        if filter_option == "ofc evolution downward":
            if patient.ofc_evolution == 3:
                return True
            return False

        if filter_option == "mri onset normal":
            if patient.mri_onset == 0:
                return True
            return False

        if filter_option == "mri onset abnormal":
            if patient.mri_onset == 1:
                return True
            return False

        if filter_option == "mri follow-up normal":
            if patient.mri_followup == 0:
                return True
            return False

        if filter_option == "mri follow-up abnormal":
            if patient.mri_followup == 1:
                return True
            return False

    # (1 : seizure free, 2 : seizure reduction, 3 : no effect, 4 : worsening, -1 no info)

    return True


main()
