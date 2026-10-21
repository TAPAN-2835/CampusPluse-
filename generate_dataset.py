"""
CampusPulse - Dataset Generation Script
Constructs a balanced, realistic corpus of ~540 student complaints across 6 categories
and a benchmark pair evaluation dataset of 120 labeled pairs for duplicate detection.
All examples are honestly documented as synthetic, high-fidelity campus complaint data.
"""

import os
import random
import pandas as pd

random.seed(42)

CATEGORIES = [
    "Hostel and Accommodation",
    "Internet and Wi-Fi",
    "Electricity and Infrastructure",
    "Water and Sanitation",
    "Academics and Administration",
    "Other"
]

LOCATIONS = [
    "Aryabhatta Hostel, Block A",
    "Aryabhatta Hostel, Block B",
    "Kalpana Chawla Girls Hostel",
    "Ramanujan Hostel, 3rd Floor",
    "Tagore Hostel, Wing C",
    "Central Library, 2nd Floor",
    "Lecture Hall Complex, Room 102",
    "Computer Science Block, Lab 4",
    "Mechanical Workshop Building",
    "Main Administrative Block",
    "Campus Cafeteria & Mess 1",
    "Sports Complex & Gymnasium",
    "Electrical Engineering Dept",
    "Biotech Research Wing",
    "Auditorium & Student Center"
]

# Diverse complaint seed patterns per category
HOSTEL_SEEDS = [
    ("Broken bed frame in room", "The wooden plank of my bed in room {room} cracked last night. Cannot sleep properly, please replace.", "Medium"),
    ("Room cupboard lock broken", "The metal latch of my wardrobe is broken and my valuables are not safe. Please send carpenter urgently.", "High"),
    ("Severe termite infestation", "Termites are destroying the study table and books in room {room}. Pest control needed immediately.", "High"),
    ("Window pane shattered", "The glass window in room {room} got cracked due to strong winds yesterday. Cold air entering at night.", "Medium"),
    ("Hostel laundry washing machine not draining", "Common laundry washing machine on 2nd floor is clogged and full of dirty water for two days.", "Low"),
    ("Geyser not working in winter", "No hot water in bathroom {room}. Geyser indicator is off and water is freezing cold.", "Medium"),
    ("Dampness and fungal patch on ceiling", "Roof ceiling in our room has continuous water seepage causing foul smell and black fungus.", "Medium"),
    ("Curtain rod fell off wall", "The curtain rod bracket in room {room} came loose from the wall and fell down.", "Low"),
    ("Room door lock jammed", "The cylindrical lock of room {room} gets stuck from outside. I almost got locked in this morning.", "High"),
    ("Mess food quality cold and unhygienic", "Dinner served at hostel mess yesterday had undercooked rice and stale dal. Several students feeling sick.", "High"),
    ("Mattress in poor condition", "The provided hostel mattress has sunken springs poking into my back. Requesting replacement.", "Low"),
    ("Cockroach menace in corridor", "Heavy cockroach infestation near the water cooler and corridor trash bins in block.", "Medium"),
    ("Ironing board damaged", "The common room iron box has a burnt cord and iron table is wobbly.", "Low"),
    ("Hostel common room TV remote missing", "Students cannot operate the TV in recreation room as buttons are stuck and remote is lost.", "Low"),
    ("Balcony pigeon netting torn", "Pigeons entering balcony and making mess everywhere because net is torn.", "Low")
]

WIFI_SEEDS = [
    ("Campus Wi-Fi authentication portal loop", "Whenever I connect to CampusNet Wi-Fi, the captive portal keeps reloading without accepting my LDAP credentials.", "High"),
    ("No Wi-Fi signal in corner rooms", "Signal strength in rooms {room} to {room_end} drops to zero bars. Constant disconnection during lectures.", "Medium"),
    ("Extreme packet loss during online tests", "Facing 40% packet loss on hostel Wi-Fi. My coding platform quiz timed out twice.", "High"),
    ("Ethernet LAN port dead in room", "The RJ45 wall jack in room {room} does not establish link. Cable tested fine on router.", "Medium"),
    ("Library Wi-Fi disconnecting every 5 minutes", "Central library reading room Wi-Fi drops connection repeatedly. Cannot download research papers.", "Medium"),
    ("Slow download speed below 200 kbps", "Internet speed in block is throttled to dial-up speeds. Unable to load LMS lectures or attend webinars.", "Medium"),
    ("DNS failure on student network", "DNS server 10.0.0.1 not resolving external domains like github and arxiv on college Wi-Fi.", "Medium"),
    ("5GHz Wi-Fi band not broadcasting", "Only 2.4GHz SSID is visible in hostel wing, causing heavy congestion and high latency.", "Low"),
    ("IP address conflict on hostel subnet", "Laptop shows 'IP conflict with another system' every time I connect to hostel access point.", "Medium"),
    ("Cannot access university ERP from outside hostel", "ERP portal page times out only when connected via campus Wi-Fi router.", "Medium"),
    ("Wi-Fi router in corridor blinking red", "The ceiling mounted access point near room {room} has a red LED and is not broadcasting any SSID.", "High"),
    ("VPN access blocked on campus network", "Unable to establish secure tunnel to university remote computing server for lab work.", "Low")
]

ELECTRICITY_SEEDS = [
    ("Sparks coming from switchboard", "Heavy electrical spark and crackling sound when plugging laptop into wall outlet in room {room}. Dangerous hazard!", "Critical"),
    ("Flickering tube light in lecture hall", "Tube light 4 in hall {room} keeps flickering violently causing eye strain during lectures.", "Low"),
    ("Ceiling fan rotating at extremely low speed", "Fan regulator in room {room} is broken and fan barely moves even at position 5 in humid heat.", "Medium"),
    ("Burning plastic smell from AC unit", "Strong burning odor coming from wall air conditioner in computer lab. Switched off breaker.", "Critical"),
    ("Complete power outage in corridor", "Entire 3rd floor corridor lights and emergency lamps are dead since yesterday evening.", "High"),
    ("Lift stuck between floors", "Elevator in academic block stopped suddenly between 2nd and 3rd floor. Alarm bell button is unresponsive.", "Critical"),
    ("Loose live wire hanging near staircase", "An uninsulated electrical cable is hanging near the 1st floor stair landing. Risk of electric shock.", "Critical"),
    ("Exhaust fan in laboratory noisy and jammed", "Chemistry research lab exhaust fan stopped working. Fumes accumulating inside lab.", "High"),
    ("MCB tripping repeatedly", "The miniature circuit breaker for wing C trips every 20 minutes when two rooms turn on study lamps.", "Medium"),
    ("Projector power supply dead", "Smart classroom projector does not power on. Power cable and socket checked.", "Medium"),
    ("Streetlight outside hostel not functioning", "Pathway between hostel gate and library is pitch dark at night due to dead pole light.", "High"),
    ("Air conditioner leaking water on server rack", "Split AC indoor unit is dripping condensate water directly onto network switch rack.", "Critical")
]

WATER_SEEDS = [
    ("No running water in washrooms", "All taps and flush tanks are completely dry on 2nd floor since morning. Urgent supply needed.", "Critical"),
    ("Brown muddy tap water from drinking cooler", "Water cooler on ground floor is dispensing yellowish contaminated water with strange odor. Students falling sick.", "Critical"),
    ("Continuous water leakage under wash basin", "Pipe joint under sink 3 in bathroom {room} is cracked and spraying water across floor.", "Medium"),
    ("Toilet flush cistern running continuously", "Flush valve stuck open causing hundreds of liters of water wastage every hour in block washroom.", "Medium"),
    ("Foul sewage odor from floor drain", "Severe sewer gas smell backing up from bathroom drainage trap into residential rooms.", "High"),
    ("Geyser leaking hot water from top valve", "Water heater in common washroom has dripping boiling water creating slip and burn hazard.", "High"),
    ("Drinking water cooler cooling compressor failed", "Water cooler in mechanical block dispensing warm water during peak summer afternoon.", "Low"),
    ("Water purifier filter replacement overdue", "TDS reading on water purifier indicates 480 PPM. Filter indicator shows red service required.", "High"),
    ("Washroom sink drainage choked", "Sink in 1st floor restroom is blocked with debris and filled with stagnant dirty water.", "Medium"),
    ("Overhead water tank overflowing", "Hostel rooftop water tank has been overflowing for three hours flooding the terrace staircase.", "Medium"),
    ("Low water pressure in showers", "Water pressure in morning hours is too weak to use showers on 4th floor.", "Low")
]

ACADEMICS_SEEDS = [
    ("Discrepancy in end-term internal marks", "Internal continuous evaluation marks for CS301 shown as 12 instead of 24 submitted by course instructor.", "High"),
    ("Exam fee receipt not generated after transaction", "Bank deducted 2500 INR for semester exam fee but portal status still shows 'Payment Pending'.", "High"),
    ("Clash in mid-semester exam timetable", "Elective Machine Learning exam and Compiler Design exam scheduled on same day and same slot.", "High"),
    ("Attendance shortfall due to medical leave not updated", "Medical certificate submitted to academic section on 5th Oct not reflected on portal; attendance blocked.", "High"),
    ("Grade card name spelling mistake", "My degree transcript preview has a typo in surname. Need official correction before convocation.", "Medium"),
    ("Library fine credited incorrectly", "Charged late fee of 350 rupees for book returned on time before due date with stamp proof.", "Low"),
    ("Bonafide certificate application pending for 10 days", "Need bonafide certificate for passport renewal urgently but admin desk has not approved request.", "Medium"),
    ("Course registration portal server error 500", "Students unable to select open electives as registration server crashes on submit.", "High"),
    ("Lab equipment missing components for practicals", "Digital electronics lab has only 4 working CROs for a batch of 40 students.", "Medium"),
    ("Faculty evaluation feedback form not accessible", "End-term faculty evaluation link in student portal is broken with 404 page not found.", "Low"),
    ("Scholarship verification documents unreviewed", "National scholarship portal verification pending at nodal college officer level.", "Medium")
]

OTHER_SEEDS = [
    ("Stray dog menace inside cafeteria", "Pack of aggressive stray dogs entering food court area and barking at students carrying trays.", "High"),
    ("Unauthorized cars parked in two-wheeler zone", "Outside visitors parking SUVs in student scooter bay blocking bike exit.", "Low"),
    ("Campus shuttle electric bus skipping scheduled stops", "Morning 8:30 AM campus bus did not stop at hostel terminal resulting in students missing class.", "Medium"),
    ("Loud construction noise during exam week", "Drilling and heavy earthmover work outside central library during quiet study hours.", "Medium"),
    ("Gymnasium treadmill cable snapped", "Cable pulley on main running machine broken. Risk of physical injury to students.", "Medium"),
    ("Lost student college ID card inquiry", "Lost my university ID card near sports ground yesterday. Security desk claims no log entry.", "Low"),
    ("Overgrown grass on football ground", "Sports field has tall wild weeds and snakes sighted near goal post perimeter.", "Medium"),
    ("Campus ATM out of cash for past weekend", "Only bank ATM on campus is out of cash and displaying out of service screen.", "Low"),
    ("Broken bench in student open park", "Concrete seating bench near fountain is cracked and tilting dangerously.", "Low"),
    ("Excessive garbage accumulated near food court", "Overflowing waste bins outside canteen attracting flies and stench.", "Medium")
]

SEED_MAPPING = {
    "Hostel and Accommodation": HOSTEL_SEEDS,
    "Internet and Wi-Fi": WIFI_SEEDS,
    "Electricity and Infrastructure": ELECTRICITY_SEEDS,
    "Water and Sanitation": WATER_SEEDS,
    "Academics and Administration": ACADEMICS_SEEDS,
    "Other": OTHER_SEEDS
}

VARIATIONS = [
    lambda text: text,
    lambda text: f"Kindly look into this: {text}",
    lambda text: f"Urgent request! {text}",
    lambda text: f"{text} Please resolve at the earliest.",
    lambda text: f"Facing this issue since yesterday. {text}",
    lambda text: f"{text} It is causing major inconvenience to all students here.",
    lambda text: f"Reporting this matter: {text} Needs quick inspection.",
    lambda text: text.replace("Please", "Pls").replace("cannot", "cant").replace("water", "wtr"),
    lambda text: f"Complaint from resident student: {text} Thanks.",
    lambda text: f"{text} Action required immediately as exams are approaching."
]


def generate_complaints_dataset(target_total: int = 540) -> pd.DataFrame:
    """Generates a balanced dataset of labeled campus complaints."""
    records = []
    per_cat = target_total // len(CATEGORIES)

    for cat in CATEGORIES:
        seeds = SEED_MAPPING[cat]
        cat_records = 0
        while cat_records < per_cat:
            seed_title, seed_desc, default_urgency = random.choice(seeds)
            var_func = random.choice(VARIATIONS)

            room_num = random.randint(101, 435)
            room_end = room_num + random.randint(2, 6)
            desc_formatted = seed_desc.replace("{room}", str(room_num)).replace("{room_end}", str(room_end))
            desc_final = var_func(desc_formatted)

            loc = random.choice(LOCATIONS)
            if "Hostel" in cat:
                loc = random.choice([l for l in LOCATIONS if "Hostel" in l])
            elif "Academics" in cat:
                loc = random.choice([l for l in LOCATIONS if "Administrative" in l or "Hall" in l or "Library" in l or "Lab" in l])
            elif "Internet" in cat:
                loc = random.choice([l for l in LOCATIONS if "Hostel" in l or "Library" in l or "Lab" in l])

            records.append({
                "title": seed_title,
                "description": desc_final,
                "category": cat,
                "location": loc,
                "urgency_hint": default_urgency
            })
            cat_records += 1

    df = pd.DataFrame(records)
    # Shuffle reproducibility
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def generate_complaint_pairs(num_pairs: int = 120) -> pd.DataFrame:
    """
    Generates high-quality test pairs for duplicate detection evaluation.
    Balanced: 60 duplicates and 60 non-duplicates.
    Features a realistic spectrum from near-identical student tickets to paraphrased reports,
    along with hard negatives (shared location/words, distinct problem) and completely distinct issues.
    """
    pairs = []

    # 1. High-overlap / Near-identical duplicates (sim ~0.70 - 0.95)
    near_duplicates = [
        ("Wi-Fi not working in room 302 Ramanujan Hostel",
         "Wi-Fi is not working in room 302 Ramanujan Hostel. Please fix."),
        ("Water leaking continuously under washbasin in 2nd floor bathroom",
         "Water is continuously leaking under the washbasin in 2nd floor bathroom"),
        ("Electrical sparks coming from wall switchboard in room 204",
         "Dangerous electrical sparks coming from wall switchboard in room 204"),
        ("Lift stuck between 2nd and 3rd floor in academic block",
         "Academic block lift is stuck between 2nd and 3rd floor"),
        ("Toilet flush tank running continuously on 3rd floor washroom",
         "Flush tank continuously running in 3rd floor washroom wasting water"),
        ("Semester exam fee deducted from bank but portal shows payment pending",
         "Exam fee deducted from bank account but university portal still shows payment pending"),
        ("Room 412 cylindrical door lock jammed, unable to enter",
         "Door lock jammed in room 412, unable to open door"),
        ("Ceiling fan regulator broken in room 115, fan running at slowest speed",
         "Fan regulator broken in room 115, ceiling fan stuck at very low speed"),
        ("Water cooler on ground floor dispensing dirty muddy water",
         "Ground floor water cooler dispensing muddy dirty water to students"),
        ("Aggressive stray dogs inside the campus cafeteria food court",
         "Stray dogs roaming inside campus cafeteria and barking at students"),
    ]
    for t1, t2 in near_duplicates:
        pairs.append({"text_1": t1, "text_2": t2, "is_duplicate": 1})
        pairs.append({"text_1": f"Urgent: {t1}", "text_2": f"Please resolve: {t2}", "is_duplicate": 1})

    # 2. Paraphrased duplicates (sim ~0.40 - 0.70)
    paraphrase_duplicates = [
        ("No internet connectivity on 3rd floor of Raman hostel, wifi drops constantly",
         "Campus wifi keeps disconnecting in Ramanujan block 3rd floor rooms"),
        ("Drinking water filter is contaminated and smells bad in hostel mess",
         "Yellow foul smelling water coming from hostel mess drinking dispenser"),
        ("Short circuit hazard with burning smell from computer lab AC unit",
         "Air conditioner in CS lab emitting burning odor, breaker switched off"),
        ("Continuous assessment marks for CS301 recorded wrongly as 12 instead of 24",
         "Discrepancy in CS301 internal midterm marks on student ERP portal"),
        ("Complete water blackout on 2nd floor, all bathroom taps bone dry",
         "Zero water supply across 2nd floor washrooms since early morning"),
        ("Bonafide certificate request pending at academic administration for two weeks",
         "Admin office has not signed my bonafide application submitted 14 days ago"),
        ("Wardrobe lock latch broken in room 308, cannot secure belongings",
         "Hostel room 308 closet lock damaged, personal luggage at risk of theft"),
        ("Campus shuttle morning bus skipped girls hostel pickup point at 8:30 AM",
         "8:30 AM electric shuttle did not stop at hostel terminal today"),
        ("Termite infestation eating away study desk and books in room 210",
         "Severe pest problem with termites destroying furniture in room 210"),
        ("Pigeon safety net torn on balcony, birds dirtying the room",
         "Balcony bird net damaged allowing pigeons to nest inside hostel room"),
    ]
    for t1, t2 in paraphrase_duplicates:
        pairs.append({"text_1": t1, "text_2": t2, "is_duplicate": 1})
        pairs.append({"text_1": f"Complaint: {t1}", "text_2": f"Report: {t2}", "is_duplicate": 1})

    # 3. Hard Negatives (Shared location/domain keywords, but completely DIFFERENT problems: sim ~0.15 - 0.40)
    hard_negatives = [
        ("Wi-Fi not working in room 302 Ramanujan Hostel",
         "Ceiling fan making loud squeaking noise in room 302 Ramanujan Hostel"),
        ("Water leaking continuously under washbasin in 2nd floor bathroom",
         "No hot water in bathroom geyser on 2nd floor"),
        ("Electrical sparks coming from wall switchboard in room 204",
         "Window glass cracked due to storm in room 204"),
        ("Lift stuck between 2nd and 3rd floor in academic block",
         "Projector power cable missing in academic block room 102"),
        ("Toilet flush tank running continuously on 3rd floor washroom",
         "Mirror cracked above washbasin in 3rd floor washroom"),
        ("Semester exam fee deducted from bank but portal shows payment pending",
         "Library fine of 350 rupees wrongfully charged for returned textbook"),
        ("Room 412 cylindrical door lock jammed, unable to enter",
         "Curtain rod detached from wall bracket in room 412"),
        ("Water cooler on ground floor dispensing dirty muddy water",
         "Ground floor fire extinguisher safety pin missing"),
        ("Aggressive stray dogs inside the campus cafeteria food court",
         "Excessive uncollected food waste overflowing near cafeteria bins"),
        ("Complete water blackout on 2nd floor, all bathroom taps bone dry",
         "Exhaust fan in 2nd floor washroom is noisy and vibrating"),
        ("Continuous assessment marks for CS301 recorded wrongly as 12 instead of 24",
         "Timetable clash between CS301 exam and open elective slot"),
        ("Short circuit hazard with burning smell from computer lab AC unit",
         "Ethernet switch port 14 in computer lab has high packet loss"),
        ("Campus shuttle morning bus skipped girls hostel pickup point",
         "Shuttle bus driver driving aggressively on campus ring road"),
        ("Wardrobe lock latch broken in room 308",
         "Damp patch and seepage on ceiling of room 308"),
        ("Drinking water filter is contaminated and smells bad in hostel mess",
         "Mess food dinner yesterday was undercooked and cold"),
    ]
    for t1, t2 in hard_negatives:
        pairs.append({"text_1": t1, "text_2": t2, "is_duplicate": 0})
        pairs.append({"text_1": f"Ticket A: {t1}", "text_2": f"Ticket B: {t2}", "is_duplicate": 0})

    # 4. Completely Distinct Negatives (Different domain, zero similarity)
    distinct_negatives = [
        ("Wi-Fi not working in room 302 Ramanujan Hostel",
         "Fee receipt not generated after online transaction for semester"),
        ("Electrical sparks coming from wall switchboard in room 204",
         "Overgrown wild grass and weeds on football sports ground"),
        ("Water cooler on ground floor dispensing dirty muddy water",
         "Lost university student ID card near gymnasium"),
        ("Lift stuck between 2nd and 3rd floor in academic block",
         "Hostel laundry washing machine full of dirty water"),
        ("Aggressive stray dogs inside the campus cafeteria food court",
         "Discrepancy in end-term internal marks on student portal"),
        ("Toilet flush tank running continuously on 3rd floor washroom",
         "ATM machine outside campus main gate out of cash"),
        ("Semester exam fee deducted from bank but portal shows payment pending",
         "Air conditioner in computer lab dripping water on floor"),
        ("Room 412 cylindrical door lock jammed, unable to enter",
         "Loud drilling and construction noise outside central library during exams"),
        ("Bonafide certificate request pending at academic administration",
         "Pigeon safety net torn on balcony allowing birds inside"),
        ("Ceiling fan regulator broken in room 115",
         "Campus shuttle electric bus skipping scheduled stops"),
    ]
    for t1, t2 in distinct_negatives:
        pairs.append({"text_1": t1, "text_2": t2, "is_duplicate": 0})
        pairs.append({"text_1": f"Notice: {t1}", "text_2": f"Inquiry: {t2}", "is_duplicate": 0})

    pairs_df = pd.DataFrame(pairs).head(num_pairs)
    pairs_df = pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return pairs_df


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Generating synthetic complaints dataset...")
    complaints_df = generate_complaints_dataset(target_total=540)
    complaints_path = os.path.join("data", "complaints.csv")
    complaints_df.to_csv(complaints_path, index=False)
    print(f"Saved {len(complaints_df)} complaints to {complaints_path}")
    print("Class distribution:")
    print(complaints_df["category"].value_counts())

    print("\nGenerating duplicate detection pair benchmark...")
    pairs_df = generate_complaint_pairs(num_pairs=120)
    pairs_path = os.path.join("data", "complaint_pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved {len(pairs_df)} pairs to {pairs_path}")
    print("Duplicate label distribution:")
    print(pairs_df["is_duplicate"].value_counts())


if __name__ == "__main__":
    main()
