from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained("ashishkgpian/betty_icd9_classifier_ehr_symptoms_text_icd9_150_epochs")
model = AutoModelForSequenceClassification.from_pretrained("ashishkgpian/betty_icd9_classifier_ehr_symptoms_text_icd9_150_epochs", ignore_mismatched_sizes=True)
model.to(device)

classes = str('403 486 582 585 425 276 710 724 458 287 285 275 583 558 327 228 338 789 790 V451 531 410 414 725 191 331 530 411 482 272 305 194 197 255 424 584 998 682 511 599 428 349 401 V100 V453 V586 041 251 E932 V300 V053 V290 571 070 250 570 572 286 518 038 280 263 995 303 244 112 881 903 955 E956 745 762 441 496 447 440 997 274 427 V104 V101 V120 V090 569 560 491 V458 433 436 493 996 416 V310 765 769 774 770 747 776 772 362 198 V103 746 766 V293 853 780 E888 730 357 430 293 443 V158 396 365 135 311 E935 721 214 437 242 600 189 304 711 800 E814 873 781 378 951 767 431 294 042 V141 V071 764 775 969 295 E950 266 779 355 553 965 E850 E853 426 804 E916 202 V502 398 707 348 787 564 V428 238 300 788 332 V107 V433 E879 861 423 E966 200 555 771 270 335 723 079 851 807 864 865 860 413 782 V108 507 512 752 162 783 778 333 785 136 799 E931 157 574 568 E878 722 719 V125 296 478 V170 805 596 E880 822 733 578 459 438 008 V098 185 967 225 V457 389 412 593 345 201 515 E933 278 492 715 415 V105 535 608 E870 V058 513 709 E821 V173 824 911 913 E812 576 203 281 580 V450 216 V340 579 693 351 088 714 E849 307 421 786 E942 959 E928 588 364 V642 V025 252 283 784 611 622 289 446 729 V498 V456 795 E854 V667 155 V130 882 852 957 E815 466 792 434 342 153 E934 481 910 456 453 867 273 532 806 V422 V541 556 394 444 924 E960 514 763 218 359 340 999 451 324 E939 537 737 455 E884 V427 591 592 577 557 575 356 368 552 500 750 253 292 E937 211 288 773 314 V652 432 379 435 E930 199 V641 494 966 758 E855 741 918 V436 078 562 820 801 839 E881 V584 731 E885 812 156 567 696 501 712 V707 215 754 753 508 876 720 V442 871 958 802 847 397 196 346 E968 510 404 360 376 370 V026 904 928 821 823 150 573 850 V497 E938 V533 V556 728 870 V874 V153 V644 V600 521 301 164 054 344 464 442 V150 282 V08 891 808 866 902 117 484 760 V048 691 519 528 320 369 685 V625 794 793 318 V441 761 936 E915 457 395 053 V113 V632 386 623 290 204 271 E819 811 813 884 E813 751 366 297 V440 473 E910 V420 057 536 152 970 485 235 372 E882 127 160 170 V880 595 909 V443 490 343 319 130 698 E823 246 854 868 872 982 151 V853 980 E980 291 517 268 487 E866 796 V452 036 354 648 701 V063 V038 227 614 533 736 942 E924 240 921 V454 977 759 768 923 E816 681 138 358 950 922 205 990 009 619 417 279 257 E860 755 991 E957 241 810 920 V461 V127 261 429 550 874 756 935 831 718 962 E858 803 480 674 277 880 879 377 529 047 083 835 462 336 E947 V160 420 317 454 E883 840 V550 960 586 933 597 350 E911 742 V614 298 V551 620 716 V462 V180 706 565 452 825 322 154 040 110 605 607 461 704 713 945 052 948 323 325 934 516 039 975 971 994 666 V111 907 E929 566 603 405 049 237 V161 V553 262 743 422 337 625 757 527 309 815 V163 402 869 E912 188 590 V852 V446 E852 886 E919 183 862 875 877 890 E944 E936 V444 598 V552 226 E818 617 E958 V123 748 968 V298 465 972 E826 905 E969 744 E829 V301 388 V146 V151 887 375 334 E848 E918 284 E876 260 987 E890 834 522 692 V588 310 863 E834 192 035 V174 171 738 220 477 212 172 V548 726 526 V099 777 749 E922 952 V320 901 542 449 V011 963 E822 524 V052 V539 144 445 321 380 604 383 587 137 845 695 V496 180 618 V102 540 525 916 174 V628 892 816 V171 520 708 176 791 V854 E906 V714 V554 V435 883 927 V434 007 581 V202 140 642 644 654 V270 V252 193 V838 V555 139 V195 V068 601 826 694 626 956 245 919 299 727 684 647 E941 V850 665 391 308 633 639 V230 V061 223 269 V183 046 534 361 673 643 986 005 034 382 239 232 V169 E901 908 634 836 616 E917 734 V698 133 E887 V445 V155 E949 142 E987 236 470 463 E940 229 448 702 182 E825 V851 814 V881 259 906 161 E891 830 E953 195 093 472 914 E988 930 543 686 900 075 705 939 381 V311 V168 018 004 917 483 656 641 217 V291 V164 E943 134 635 659 E920 506 E869 111 096 094 123 158 141 243 690 097 632 989 964 027 V596 373 V017 254 932 187 353 669 V504 602 843 912 374 983 E864 031 210 114 646 077 V018 670 615 V638 V135 938 V580 680 878 E965 471 652 663 658 V272 213 032 148 V643 V148 V062 E989 E927 131 233 V040 V066 125 V503 V581 V292 V192 700 703 209 V029 208 697 E871 184 015 146 V140 V154 992 249 149 V142 844 175 V542 363 V152 V106 V688 V265 012 885 E955 V530 385 V124 V741 390 474 627 817 230 E817 V198 E862 258 V463 735 V024 V640 976 E861 V765 V023 V626 E828 V188 341 V560 798 V448 893 495 084 523 V653 953 V549 V095 V182 621 475 V425 058 306 V165 551 E831 V136 V109 256 219 221 961 985 828 671 E820 897 V840 926 V421 048 594 896 082 E986 541 145 267 683 V097 732 265 011 E801 V185 664 V620 E840 V166 V468 629 115 V587 E908 120 V708 098 V469 V694 E824 E970 121 838 832 460 013 V239 944 V189 946 118 326 E945 645 352 159 E967 V618 147 V908 941 312 624 V186 V145 661 010 E865 091 E886 649 E905 E962 V612 E959 502 V438 V222 163 947 V162 E946 V716 315 367 V540 846 717 V561 V175 842 V138 V703 V583 841 672 062 488 347 339 E841 086 V400 E985 655 974 V289 V604 V074 V728 371 190 V126 090 143 943 V611 V331 085 V172 E835 668 740 V167 V558 E851 E811 V430 837 V072 V431 302 E923 V110 E900 V562 E963 E964 V118 V624 E800 988 833 023 V020 021 003 V660 E806 313 E954 V860 660 V449 231 V602 186 E863 E874 V721 V181 651 033 V654 E804 330 610 384 E838 E001 973 819 014 132 E899 925 207 V861 E002 E030 E000 894 E873 E999 E976 E003 V016 E805 045 V610 V078 V510 E029 848 E006 V403 122 V536 E013 E019 173 E913 677 E008 V568 V143 V091 V872 066 V601 116 V882 V065 538 V655 316 E007 E016 E921 V902 206 V254 099 V489 V870 E977 628 V250 E982 V486 539 V073 937 V812 030 V271 589 V672 V671 E926 E925 E857 V537 954 E827 657 V910 V789 V037 E975 V045 V848 393 V426 179 387 V903 E856 V901 915').split(' ')

# print(classes)

from neo4j import GraphDatabase
from transformers import pipeline
import pandas as pd

# Connect to Neo4j
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def query_kg(self, symptoms, true_length):
        """
        Query the KG for diseases related to the given symptoms.
        """
        # query = """MATCH (s:Symptom)-[:RELATED_TO]-(d:Disease)
        #         WHERE any(symptom IN $symptoms WHERE symptom IN [s.name] + s.synonyms)
        #         RETURN d.id AS disease
        #         LIMIT $true_length"""


        query = """
        MATCH (s:Symptom)-[r:HAS_SYMPTOM]-(d:Disease)
    WHERE s.name IN $symptoms OR any(synonym IN s.synonyms WHERE synonym IN $symptoms)
    RETURN d.id AS disease, d.name as name, r.weight AS weight
    ORDER BY weight DESC
    LIMIT $true_length
        """
        with self.driver.session() as session:
            result = session.run(query, symptoms=symptoms, true_length=true_length)
            # return [{"disease": record["disease"]} for record in result]
            return [{"disease": record["disease"], "name": record["name"], "weight": record["weight"]} for record in result]

neo4j_handler = Neo4jHandler("bolt://localhost:7687", "neo4j", "sparse_kg_2_123")


def get_predictions_with_rag(input_text, symptoms, true_length, threshold=0.8):
    """
    Get predictions from the model and refine them using the knowledge graph.
    """
    # Query the KG for related diseases
    kg_results = neo4j_handler.query_kg(symptoms, true_length)

    augmented_input = f"{input_text} {symptoms} {kg_results}"

    # Tokenize and predict using the model
    tokenized_input = tokenizer(
        augmented_input,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    output = model(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > threshold).nonzero()[:, 1].tolist()]
    
    
    return predicted_labels[:true_length]


# Function to calculate precision, recall, and F1 score based on predicted and true disease codes
def calculate_f1(true_codes, predicted_codes):
    true_prefixes = {code[:3] for code in true_codes}
    pred_prefixes = {str(code)[:3] for code in predicted_codes}

    # True Positives (TP): Codes correctly predicted
    true_positives = len(true_prefixes & pred_prefixes)
    
    # False Positives (FP): Predicted codes that are not in true codes
    false_positives = len(pred_prefixes - true_prefixes)
    
    # False Negatives (FN): True codes that were not predicted
    false_negatives = len(true_prefixes - pred_prefixes)
    
    # Calculate Precision and Recall
    precision = true_positives / len(predicted_codes)
    recall = true_positives / len(true_codes)
    
    # Calculate F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


import csv, ast
import pandas as pd
raw_test_df = pd.read_csv('symptoms_test.csv')
raw_test_df = raw_test_df.drop('Unnamed: 0',axis =1)

from tqdm import tqdm
true_labels = []
predicted_labels = []
# Wrap raw_test_df.iterrows() with tqdm
output_file_path = "bettywithkg_frmk1.csv"
with open(output_file_path, mode='w', newline='', encoding="utf-8") as outfile:
    fieldnames = ["symptoms", "true_disease_codes", "predicted_diseases", "precision", "recall", "f1_score"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    # Generate predictions and calculate F1 scores
    f1_scores = []
    prec = []
    rec = []
    for i, j in tqdm(raw_test_df.iterrows(), total=len(raw_test_df)):
        symptom_list = ast.literal_eval(j["Symptoms"])
        true_label = j.short_codes.split(',')
        pred_label = get_predictions_with_rag(str(j.text) + ' ' + str(j.Symptoms), symptom_list, len(true_label), 0.0643745388060965)
        # pred_label = get_predictions(str(j.text) + ' ' + str(j.Symptoms), 0.0643745388060965)
        
        true_labels.append(true_label)
        predicted_labels.append(pred_label)


        
        # Calculate precision, recall, and F1 score
        precision, recall, f1 = calculate_f1(true_label, pred_label)
        f1_scores.append(f1)
        prec.append(precision)
        rec.append(recall)
        
        # Write the results for this sample to the CSV file
        writer.writerow({
            "symptoms": str(j.Symptoms),
            "true_disease_codes": ", ".join(true_label),
            "predicted_diseases": ", ".join(map(str, pred_label)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

macro_f1_score = sum(f1_scores) / len(f1_scores)
macro_prec = sum(prec) / len(prec)
macro_rec = sum(rec) / len(rec)
print(f"Macro Precision Score: {macro_prec:.4f}")
print(f"Macro Recall Score: {macro_rec:.4f}")
print(f"Macro F1 Score: {macro_f1_score:.4f}")
