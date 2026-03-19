#!/usr/bin/env python3
"""
Save scraped gap-fill pages into the knowledge base.
Run from project root: venv/bin/python3 scripts/save_gap_pages.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scraping.scrape_mass_gov_browser import save_browser_page

# ─── Batch 1: Regulations ─────────────────────────────────────────────────────

pages = []

# 940 CMR 3.17 - Landlord-Tenant Consumer Protection Regulations
pages.append(dict(
    url="https://www.law.cornell.edu/regulations/massachusetts/940-CMR-3-17",
    title="940 CMR 3.17 - Landlord-Tenant Consumer Protection Regulations",
    content_type="regulation",
    crawl_depth=1,
    parent_url="https://www.law.cornell.edu/regulations/massachusetts/940-CMR-3-17",
    text="""940 CMR § 3.17 - Landlord-Tenant

(1) Conditions and Maintenance of a Dwelling Unit. It shall be an unfair or deceptive act or practice for an owner to:

(a) Rent a dwelling unit which, at the inception of the tenancy:
1. contains a condition which amounts to a violation of law which may endanger or materially impair the health, safety, or well-being of the occupant; or
2. is unfit for human habitation;

(b) Fail, during the terms of the tenancy, after notice is provided in accordance with M.G.L. c. 111, s. 127L, to:
1. remedy a violation of law in a dwelling unit which may endanger or materially impair the health, safety, or well-being of the occupant, or
2. maintain the dwelling unit in a condition fit for human habitation;
provided, however, that said violation of law was not caused by the occupant or others lawfully upon said dwelling unit;

(c) Fail to disclose to a prospective tenant the existence of any condition amounting to a violation of law within the dwelling unit of which the owner had knowledge or upon reasonable inspection could have acquired such knowledge at the commencement of the tenancy;

(d) Represent to a prospective tenant that a dwelling unit meets all requirements of law when, in fact, it contains violations of law;

(e) Fail within a reasonable time after receipt of notice from the tenant to make repairs in accordance with a pre-existing representation made to the tenant;

(f) Fail to provide services and/or supplies after the making of any representation or agreement, that such services would be provided during the term or any portion of the term of the tenancy agreement;

(g) Fail to reimburse the tenant within a reasonable or agreed time after notice, for the reasonable cost of repairs made or paid for, or supplies or services purchased by the tenant after any representation, that such reimbursement would be made;

(h) Fail to reimburse an occupant for reasonable sums expended to correct violations of law in a dwelling unit if the owner failed to make such corrections pursuant to the provisions of M.G.L. c. 111, s. 127L, or after notice prescribed by an applicable law;

(i) Fail to comply with the State Sanitary Code or any other law applicable to the conditions of a dwelling unit within a reasonable time after notice of a violation of such code or law from the tenant or agency.

(2) Notices and Demands. It shall be an unfair or deceptive practice for an owner to:

(a) Send to a tenant any notice or paper which appears or purports to be an official or judicial document but which he knows is not;

(b) Fail or refuse to accept any notice sent to any address to which rent is customarily sent, or given to any person who customarily accepts on behalf of the owner, or sent to the person designated in the rental agreement in accordance with 940 CMR 3.17(3)(b)2.

(c) Demand payment for increased real estate taxes during the term of the tenancy unless, prior to the inception of the tenancy, a valid agreement is made pursuant to which the tenant is obligated to pay such increase.

(3) Rental Agreements.

(a) It shall be unfair or deceptive act or practice for an owner to include in any rental agreement any term which:
1. Violates any law;
2. Fails to state clearly and conspicuously in the rental agreement the conditions upon which an automatic increase in rent shall be determined. Provided, however, that nothing contained in 940 CMR 3.17(3)(a)2. shall be deemed to invalidate an otherwise valid tax escalator clause;
3. Contains a penalty clause not in conformity with the provisions of M.G.L. c. 186, s. 15B;
4. Contains a tax escalator clause not in conformity with the provisions of M.G.L. c. 186, s. 15C;

(b) It shall be an unfair or deceptive practice for an owner to enter into a written rental agreement which fails to state fully and conspicuously, in simple and readily understandable language:
1. The names, addresses, and telephone numbers of the owner, and any other person who is responsible for the care, maintenance and repair of the property;
2. The name, address, and telephone number of the person authorized to receive notices of violations of law and to accept service of process on behalf of the owner;
3. The amount of the security deposit, if any; and that the owner must hold the security deposit in a separate, interest-bearing account and give to the tenant a receipt and notice of the bank and account number; that the owner must pay interest, at the end of each year of the tenancy, if the security deposit is held for one year or longer from the commencement of the tenancy; that the owner must submit to the tenant a separate written statement of the present condition of the premises, as required by law, and that, if the tenant disagrees with the owner's statement of condition, he/she must attach a separate list of any damage existing in the premises and return the statement to the owner; that the owner must, within thirty days after the end of the tenancy, return to the tenant the security deposit, with interest, less lawful deductions as provided in M.G.L. c. 186, s. 15B; that if the owner deducts for damage to the premises, the owner shall provide to the tenant, an itemized list of such damage, and written evidence indicating the actual or estimated cost of repairs necessary to correct such damage; that no amount shall be deducted from the security deposit for any damage which was listed in the separate written statement of present condition or any damage listed in any separate list submitted by the tenant and signed by the owner or his agent; that, if the owner transfers the tenant's dwelling unit, the owner shall transfer the security deposit, with any accrued interest, to the owner's successor in interest for the benefit of the tenant.

(c) It shall be unfair and deceptive practice for an owner to fail to give the tenant an executed copy of any written rental agreement within 30 days of obtaining the signature of the tenant thereon.

(4) Security Deposits and Rent in Advance. It shall be an unfair or deceptive practice for an owner to:

(a) require a tenant or prospective tenant, at or prior to the commencement of any tenancy, to pay any amount in excess of the following:
1. rent for the first full month of occupancy; and
2. rent for the last full month of occupancy calculated at the same rate as the first month; and
3. a security deposit equal to the first month's rent; and
4. the purchase and installation cost for a key and lock.
or, at any time subsequent to the commencement of a tenancy, demand rent in advance in excess of the current month's rent or a security deposit in excess of the amount allowed by 940 CMR 3.17(4)(a)3.

(b) fail to give to the tenant a written receipt indicating the amount of rent in advance for the last month of occupancy, and a written receipt indicating the amount of the security deposit, if any, paid by the tenant, in accordance with M.G.L. c. 186, s. 15B;

(c) fail to pay interest at the end of each year of the tenancy, on any security deposit held for a period of one year or longer from the commencement of the term of the tenancy, as required by M.G.L. c. 186, s. 15B;

(d) fail to hold a security deposit in a separate interest-bearing account or provide notice to the tenant of the bank and account number, in accordance with M.G.L. c. 186, s. 15B;

(e) fail to submit to the tenant upon receiving a security deposit or within ten days after commencement of the tenancy, whichever is later, a separate written statement of the present condition of the premises in accordance with M.G.L. c. 186, s. 15B;

(f) fail to furnish to the tenant, within 30 days after the termination of occupancy under a tenancy-at-will or the end of the tenancy as specified in a valid written rental agreement, an itemized list of damage, if any, and written evidence indicating the actual or estimated cost of repairs necessary to correct such damage, in accordance with M.G.L. c. 186, s. 15B;

(g) fail to return to the tenant the security deposit or balance thereof to which the tenant is entitled after deducting any sums in accordance with M.G.L. c. 186, s. 15B, together with interest, within thirty days after termination of occupancy under a tenancy-at-will agreement or the end of the tenancy as specified in a valid written rental agreement;

(h) deduct from a security deposit for any damage which was listed in the separate written statement of present condition given to the tenant prior to execution of the rental agreement or creation of the tenancy, or any damages listed in any separate list submitted by the tenant and signed by the owner or his agent;

(i) fail, upon transfer of his interest in a dwelling unit for which a security deposit is held, to transfer such security deposit together with any accrued interest for the benefit of the tenant to his successor in interest, in accordance with M.G.L. c. 186, s. 15B;

(j) fail, upon transfer to him of a dwelling unit for which a security deposit is held, to assume liability for the retention and return of such security deposit, regardless of whether the security deposit was, in fact, transferred to him by the transferor of the dwelling unit, in accordance with M.G.L. c. 186, s. 15B; provided, that 940 CMR 3.17(4)(j) shall not apply to a city or town which acquires property pursuant to M.G.L. c. 60 or to a foreclosing mortgagee or a mortgagee in possession which is a financial institution chartered by the Commonwealth or the United States, or;

(k) otherwise fail to comply with the provisions of M.G.L. c. 186, s. 15B. 940 CMR 3.00 shall not be deemed to limit any rights or remedies of any tenant or other person under M.G.L. c. 186, s. 15B(6) or (7).

(5) Evictions and Termination of Tenancy. It shall be an unfair and deceptive practice for an owner to:

(a) Deprive a tenant of access to or full use of the dwelling unit or otherwise exclude him without first obtaining a valid writ of execution for possession of the premises as set forth in M.G.L. c. 239 or such other proceedings authorized by law;

(b) Commence summary process for possession of a dwelling unit before the time period designated in the notice to quit under M.G.L. c. 186, s. 11 and 12, has expired; provided, however, nothing in 940 CMR 3.17 shall effect the rights and remedies contained in M.G.L. c. 239 s.1A.

(6) Miscellaneous. It shall be an unfair and deceptive practice for an owner to:

(a) Impose any interest or penalty for late payment of rent unless such payment is 30 days overdue;

(b) Retaliate or threaten to retaliate in any manner against a tenant for exercising or attempting to exercise any legal rights as set forth in M.G.L. c. 186, s. 18;

(c) Retain as damages for a tenant's breach of lease, or the failure of a prospective tenant to enter into a written rental agreement after signing a rental application, any amount which exceeds the damages to which he is entitled under the law, or an amount which the parties have otherwise agreed as to the amount of the damages;

(d) Require payment for rent for periods during which the tenant was not obligated to occupy and did not in fact occupy the dwelling unit unless otherwise agreed to in writing by the parties;

(e) Enter a dwelling unit other than (i) to inspect the premises, or (ii) to make repairs thereto, or (iii) to show the same to a prospective tenant, purchaser, mortgagee or its agents, or (iv) pursuant to a court order, or (v) if the premises appear to have been abandoned by the tenant, or (vi) to inspect, during the last 30 days of the tenancy or after either party has given notice to the other of intention to terminate the tenancy, for the purpose of determining the amount of damage, if any, to the premises which would be cause of reduction from any security deposit held by the owner.

(f) To violate willfully any provisions of M.G.L. c. 186, § 14.

(g) It shall be an unfair practice for any owner who is obligated by law or by the express or implied terms of any tenancy agreement to provide gas or electric service to an occupant:
1. To fail to provide such service; or
2. To expose such occupant to the risk of loss of such service by failing to pay gas or electric bills when they become due or by committing larceny or unauthorized use of such gas or electricity.

Amended by Mass Register Issue 1420, eff. 3/20/2020.
""",
))

# 105 CMR 410 - State Sanitary Code (TOC + key sections combined)
pages.append(dict(
    url="https://www.law.cornell.edu/regulations/massachusetts/department-105-CMR/title-105-CMR-410.000",
    title="105 CMR 410 - State Sanitary Code Chapter II: Minimum Standards of Fitness for Human Habitation",
    content_type="regulation",
    crawl_depth=1,
    parent_url="https://www.mass.gov/regulations/105-CMR-41000-minimum-standards-of-fitness-for-human-habitation-state-sanitary-code-chapter-ii",
    text="""105 CMR 410.000 - Minimum Standards of Fitness for Human Habitation (State Sanitary Code, Chapter II)

Establishes minimum standards for housing to protect the health, safety, and well-being of occupants and the general public; provides enforcement procedures for boards of health to ensure compliance; and facilitates the use of legal remedies available to occupants of substandard housing.

Regulatory Authority: MGL c. 111, §§ 3 and 127A.

## Key Sections

### 410.180 - Temperature Requirements

(A) The owner shall provide heat in every habitable room and every room containing a toilet, shower, or bathtub from September 15th through May 31st so that it shall be:
(1) At least 68°F (20°C) between 7:00 A.M. and 11:00 P.M.; and
(2) At least 64°F (17°C) between 11:01 P.M. and 6:59 A.M.

(B) At no time shall the heating system, required by 105 CMR 410.160(A), used during the heating season cause the temperature to exceed 78°F (25°C) in any room.

(C) The temperature shall be measured at a height of five feet above floor level on a wall any point more than five feet from the exterior wall.

(D) A board of health may alter the heating season, as defined in 105 CMR 410.180(A), by ending it no earlier than May 15th or delaying the start date no later than September 30th in a particular year for all residences within its jurisdiction.

### 410.500 - Owner's Responsibility to Maintain Building and Structural Elements

(A) Every owner of a residence shall maintain all buildings and structural elements in compliance with accepted standards so they are in good repair and in every way fit for the intended use, including:
(1) Protected from wind, rain and snow, and are watertight, free from excess moisture or the appearance of mold, and pest resistant; and
(2) Free from holes, cracks, loose plaster, or defects that render the area difficult to keep clean, create an injury risk, or provide an entry or harborage for pests.

(B) In the event of leaks and flooding, the owner shall ensure all surfaces have been dried within 48 hours from the time they are notified or the end of the event, whichever is sooner.

### 410.550 - Extermination of Insects, Rodents and Skunks

Owners and occupants are responsible for the control and elimination of pests as follows:

(A) Residences containing one dwelling unit:
(1) The occupant shall maintain the unit free from all pest infestation and shall be responsible for pest elimination.
(2) The owner shall be responsible for pest elimination if they have not maintained structural or other building elements necessary to keep pests from entering.

(B) The owner of a residence containing two or more dwelling units, including a homeless shelter or a rooming house, shall maintain it and its premises free from infestation and shall be responsible for pest elimination.

(C) Extermination shall be accomplished by:
(1) Eliminating the harborage places of insects and rodents;
(2) Removing or making inaccessible materials that may serve as food or breeding ground for pests;
(3) Poisoning, spraying, fumigating, trapping pests; or
(4) Any other recognized and legal pest elimination method.

(D) All use of pesticides within the interior of a residence shall be in accordance with applicable laws and regulations. Pesticide applicators or their employers must give at least 48 hours pre-notification to occupants of all residential units prior to any commercial application of pesticides for the control of indoor household or structural indoor pests.

(F) The owner of a residence, except for a homeless shelter, shall conduct an inspection of each unit prior to a new occupancy to identify the presence of pests.

### 410.630 - Conditions Deemed to Endanger or Materially Impair Health or Safety

(A) The following conditions, when found to exist in a residence, shall always be deemed conditions which may endanger or materially impair the health, or safety and well-being of a person or persons occupying the premises:

(1) Failure to provide and maintain a supply of water sufficient in quantity, pressure and temperature, both hot and cold, for a period of 24 hours or longer.
(2) Failure to provide heat as required by 105 CMR 410.180 or improper venting or use of a space heater or water heater.
(3) Shutoff and/or failure to restore electricity, gas, or water.
(4) Failure to provide the electrical facilities required by 105 CMR 410.300(A) through (E).
(5) Failure to provide a safe supply of water.
(6) Failure to provide a toilet, sink, shower or bathtub and maintain a sewage disposal system in operable condition.
(7) Failure to provide and maintain: a kitchen sink of sufficient size; if supplied by the owner, a conventional cooktop and oven, or a refrigerator with freezer.
(8) Failure to provide and maintain adequate exits, or the obstruction of any exit, passageway or common area.
(9) Failure to comply with the security requirements of 105 CMR 410.270(A).
(10) Accumulation of refuse, filth or other causes of sickness which may provide a food source or harborage for rodents, insects or other pests.
(11) The presence of lead based paint in violation of 105 CMR 460.000.
(12) Roof, foundation, or other structural defects that may expose the occupant to fire, burns, shock, accident or other dangers.
(13) Failure to install or maintain electrical, plumbing, heating and gas-burning facilities in compliance with accepted standards.
(14) Any defect in asbestos material which may result in the release of asbestos dust.
(15) Failure to provide a smoke detector or carbon monoxide alarm.
(16) Failure to provide and maintain a railing or guard for every stairway, porch, balcony, roof or similar place.
(17) Failure to maintain the premises free from pests.
(18) Any other violation not enumerated shall be deemed a condition which may endanger health or safety upon the failure of the owner to remedy within the time ordered by the board of health.

### 410.640 - Time Frames for Correction of Violations

(A) If an inspection reveals that a residence does not comply with 105 CMR 410.000, the board of health shall:
(1) Within 12 hours after the inspection, order the owner or occupant to make a good faith effort to correct within 24 hours of service any of the violations cited in 105 CMR 410.630(A).
(2) Within seven calendar days after the inspection, order the owner or occupant to correct, within 30 calendar days of service, any violations not listed in 105 CMR 410.630(A).

(B) No order shall exceed 30 calendar days for the correction of violations, nor shall the time frames listed in the original order be extended beyond the original date, unless a hearing has been conducted.

## Table of Contents (Additional Sections)

- 410.100 Kitchen Facilities
- 410.110 Bathroom Facilities: Sinks, Toilets, Tubs, and Showers
- 410.120 Approved Toilets
- 410.130 Potable Water/Sanitary Drainage
- 410.140 Plumbing Connections
- 410.150 Hot Water
- 410.160 Heating Systems
- 410.170 Venting
- 410.200 Provision and Metering of Electricity or Gas
- 410.220 Natural and Mechanical Ventilation
- 410.235 Owner's Installation, Maintenance and Repair Responsibilities
- 410.240 Occupant's Installation and Maintenance Responsibilities
- 410.250 Asbestos-containing Material
- 410.270 Locks
- 410.330 Smoke Detectors and Carbon Monoxide Alarms
- 410.400 Owner/Manager Contact Information and Notice of Occupants' Legal Rights
- 410.420 Habitability Requirements
- 410.450 Means of Egress
- 410.460 Homeless Shelters
- 410.470 Lead-based Paint Hazards
- 410.480 Locks
- 410.501 Weathertight Elements
- 410.502 Use of Lead Paint Prohibited
- 410.560 Refuse
- 410.570 Maintenance of Areas in a Sanitary and Safe Condition
- 410.600 Inspection upon Request
- 410.650 Residences Unfit for Human Habitation
- 410.700 Variances
- 410.750 Conditions Deemed to Endanger or Impair Health or Safety
- 410.800-410.860 Hearing Procedures
- 410.900-410.950 Penalties and Enforcement

Amended by Mass Register Issue 1495, eff. 5/12/2023.
""",
))

# ─── Batch 2: MGL c.239 Eviction Sections ─────────────────────────────────────

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section1",
    title="MGL c.239 s.1 - Persons Entitled to Summary Process",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239",
    text="""MGL Chapter 239, Section 1: Persons entitled to summary process

Section 1. If a forcible entry into land or tenements has been made, if a peaceable entry has been made and the possession is unlawfully held by force, if the lessee of land or tenements or a person holding under him holds possession without right after the determination of a lease by its own limitation or by notice to quit or otherwise, or if a mortgage of land has been foreclosed by a sale under a power therein contained or otherwise, or if a person has acquired title to land or tenements by purchase, and the seller or any person holding under him refuses to surrender possession thereof to the buyer, or if a tax title has been foreclosed by decree of the land court, or if a purchaser, under a written agreement to purchase, is in possession of land or tenements beyond the date of the agreement without taking title to said land as called for by said agreement, the person entitled to the land or tenements may recover possession thereof under this chapter.

A person in whose favor the land court has entered a decree for confirmation and registration of his title to land may in like manner recover possession thereof, except where the person in possession or any person under whom he claims has erected buildings or improvements on the land, and the land has been actually held and possessed by him or those under whom he claims for six years next before the date of said decree or was held at the date of said decree under a title which he had reason to believe good.
""",
))

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section1A",
    title="MGL c.239 s.1A - Conditions and Restrictions on Residential Eviction",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239",
    text="""MGL Chapter 239, Section 1A: Land or tenements used for residential purposes; action by lessor under this chapter to recover possession; conditions and restrictions

Section 1A. A lessor of land or tenements used for residential purposes may bring an action under this chapter to recover possession thereof before the determination of the lease by its own limitation, subject to the following conditions and restrictions.

The tenancy of the premises at issue shall have been created for at least six months duration by a written lease in which a specific termination date is designated, a copy of which, signed by all parties, shall be annexed to the summons.

No such action may be initiated before the latest date permitted by the lease for either party to notify the other of his intention to renew or extend the rental agreement, or in any case before thirty days before the designated termination date of the tenancy.

The person bringing the action shall notify all defendants by registered mail that he has done so, which notification shall be mailed not later than twenty-four hours after the action is initiated.

The person bringing the action shall demonstrate substantial grounds upon which the court could reasonably conclude that the defendant is likely to continue in possession of the premises at issue without right after the designated termination date, which grounds shall be set forth in the writ.

No execution for possession may issue in any such action before the day next following the designated termination date of the tenancy.

Any action brought pursuant to this section shall conform to and be governed by the provisions of this chapter in all other respects and no remedy or procedure otherwise available to any party, including any stay of execution which the court has discretion to allow, shall be denied solely because the action was brought pursuant to this section.
""",
))

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section2A",
    title="MGL c.239 s.2A - Anti-Reprisal Defense in Eviction",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239",
    text="""MGL Chapter 239, Section 2A: Reprisal for reporting violations of law, for tenant's union activity, or actions taken pursuant to laws protecting tenants who are victims of domestic violence, rape, sexual assault or stalking; defense; presumption

Section 2A. It shall be a defense to an action for summary process that such action or the preceding action of terminating the tenant's tenancy, was taken against the tenant for the tenant's act of:
- commencing, proceeding with, or obtaining relief in any judicial or administrative action the purpose of which action was to obtain damages under or otherwise enforce, any federal, state or local law, regulation, by-law, or ordinance, which has as its objective the regulation of residential premises;
- exercising rights pursuant to section one hundred and twenty-four D of chapter one hundred and sixty-four;
- reporting a violation or suspected violation of law as provided in section eighteen of chapter one hundred and eighty-six;
- organizing or joining a tenants' union or similar organization;
- making, or expressing an intention to make, a payment of rent to an organization of unit owners pursuant to paragraph (c) of section six of chapter one hundred and 83A;
- a tenant, co-tenant or a member of the household taking action under section 3 of chapter 209A or section 3 of chapter 258E;
- seeking relief under sections 23 to 29, inclusive, of chapter 186;
- reporting to any police officer or law enforcement professional an incident of domestic violence, rape, sexual assault or stalking, as defined in said section 23 of said chapter 186, against a tenant, co-tenant or member of the household;
- reporting to any police officer or law enforcement professional a violation of a restraining order or any act of abuse or harassment directed against the tenant, co-tenant or member of the household.

The commencement of such action against a tenant, or the sending of a notice to quit upon which the summary process action is based, or the sending of a notice, or performing any act, the purpose of which is to materially alter the terms of the tenancy, within six months after the tenant has engaged in any of the above protected activities, shall create a rebuttable presumption that such summary process action is a reprisal against the tenant.

Such presumption may be rebutted only by clear and convincing evidence that such action was not a reprisal against the tenant and that the plaintiff had sufficient independent justification for taking such action, and would have in fact taken such action, in the same manner and at the same time the action was taken, even if the tenant had not commenced any legal action, made such report or engaged in such activity.
""",
))

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239/Section8A",
    title="MGL c.239 s.8A - Rent Withholding for Code Violations",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartIII/TitleIII/Chapter239",
    text="""MGL Chapter 239, Section 8A: Rent withholding; grounds; amount claimed; presumptions and burden of proof; procedures

Section 8A. In any action under this chapter to recover possession of any premises rented or leased for dwelling purposes, brought pursuant to a notice to quit for nonpayment of rent, or where the tenancy has been terminated without fault of the tenant or occupant, the tenant or occupant shall be entitled to raise, by defense or counterclaim, any claim against the plaintiff relating to or arising out of such property, rental, tenancy, or occupancy for breach of warranty, for a breach of any material provision of the rental agreement, or for a violation of any other law.

The amounts which the tenant or occupant may claim hereunder shall include, but shall not be limited to, the difference between the agreed upon rent and the fair value of the use and occupation of the premises, and any amounts reasonably spent by the tenant or occupant pursuant to section one hundred and twenty-seven L of chapter one hundred and eleven and such other damages as may be authorized by any law having as its objective the regulation of residential premises.

Whenever any counterclaim or claim of defense under this section is based on any allegation concerning the condition of the premises or the services or equipment provided therein, the tenant or occupant shall not be entitled to relief under this section unless:

(1) the owner or his agents knew of such conditions before the tenant or occupant was in arrears in his rent;

(2) the plaintiff does not show that such conditions were caused by the tenant or occupant or any other person acting under his control; except that the defendant shall have the burden of proving that any violation appearing solely within that portion of the premises under his control and not by its nature reasonably attributable to any action or failure to act of the plaintiff was not so caused;

(3) the premises are not situated in a hotel or motel, nor in a lodging house or rooming house wherein the occupant has maintained such occupancy for less than three consecutive months; and

(4) the plaintiff does not show that the conditions complained of cannot be remedied without the premises being vacated.

Proof that the premises are in violation of the standard of fitness for human habitation established under the state sanitary code, the state building code, or any other ordinance, by-law, rule or regulation establishing such standards and that such conditions may endanger or materially impair the health, safety or well-being of a person occupying the premises shall create a presumption that conditions existed in the premises entitling the tenant or occupant to a counterclaim or defense under this section.

Proof of written notice to the owner of an inspection of the premises, issued by the board of health, shall create a presumption that on the date such notice was received, such person knew of the conditions revealed by such inspection.

There shall be no recovery of possession pursuant to this chapter pending final disposition of the plaintiff's action if the court finds that the requirements of the second paragraph have been met.

The court after hearing the case may require the tenant or occupant claiming under this section to pay to the clerk of the court the fair value of the use and occupation of the premises less the amount awarded the tenant or occupant for any claim under this section, or to make a deposit with the clerk of such amount or such installments thereof from time to time as the court may direct.

There shall be no recovery of possession under this chapter if the amount found by the court to be due the landlord equals or is less than the amount found to be due the tenant or occupant by reason of any counterclaim or defense under this section.

If the amount found to be due the landlord exceeds the amount found to be due the tenant or occupant, there shall be no recovery of possession if the tenant or occupant, within one week after having received written notice from the court of the balance due, pays to the clerk the balance due the landlord, together with interest and costs of suit.

Any provision of any rental agreement purporting to waive the provisions of this section shall be deemed to be against public policy and void.

The provisions of section two A and of section eighteen of chapter one hundred and eighty-six shall apply to any tenant or occupant who invokes the provisions of this section.
""",
))

# ─── Batch 3: Fair Housing & Anti-Discrimination ──────────────────────────────

pages.append(dict(
    url="https://www.mass.gov/info-details/overview-of-fair-housing-law",
    title="Overview of Fair Housing Law - Massachusetts Attorney General",
    content_type="guide",
    crawl_depth=1,
    parent_url="https://www.mass.gov/info-details/overview-of-fair-housing-law",
    text="""Overview of Fair Housing Law

State and federal law prohibit discrimination in the sale and rental of housing by property owners, landlords, property managers, mortgage lenders, and real estate agents.

## Protected Classes in Massachusetts

In Massachusetts, it is unlawful for a housing provider to discriminate against a current or prospective tenant based on:
- Race
- Color
- National Origin
- Religion
- Sex
- Familial Status (i.e. children)
- Disability
- Source of Income (e.g. a Section 8 voucher)
- Sexual Orientation
- Gender Identity
- Age
- Marital Status
- Veteran or Active Military Status
- Genetic Information

## Examples of Fair Housing Violations

Examples of unlawful practices include:

- Refusing to rent you, or charging you higher rent or other fees, based on one of these protected characteristics.
- Steering you away from particular properties or rental units based on one of these protected characteristics.
- Refusing to rent to you because you rely on public assistance (for example, a Section 8 voucher).
- Failing or refusing to make reasonable accommodations for tenants with disabilities, including exceptions to policies (for example, a "no pets" policy) or reasonable physical modifications (grab bars or wheelchair ramps, for example).
- Harassing you, whether based on gender or any protected characteristic listed above.
- Refusing to give you a mortgage, or charging you higher fees, based on any of the protected characteristics listed above.
- Threatening to report you to immigration authorities so that you or your family members will be afraid to exercise any of your rights under the law.
- Refusing to rent to a pregnant woman or a family with young children, or evicting families, because a property contains lead paint.
- Retaliating against you if you report discrimination.

## Filing a Complaint

If you have been denied housing, charged a higher amount of rent or fees, subjected to harassment, or otherwise treated unfairly by a housing provider because of one of the characteristics listed above, you should file a complaint with:
- The Attorney General's Civil Rights Division: (617) 963-2917
- The Massachusetts Commission Against Discrimination (MCAD)
""",
))

pages.append(dict(
    url="https://www.mass.gov/info-details/massachusetts-law-about-discrimination",
    title="Massachusetts Law About Discrimination - Law Library Guide",
    content_type="guide",
    crawl_depth=1,
    parent_url="https://www.mass.gov/info-details/massachusetts-law-about-discrimination",
    text="""Massachusetts Law About Discrimination

Laws, regulations, cases and web sources on discrimination law in Massachusetts.

## Key Massachusetts Anti-Discrimination Laws

- MGL c. 4, § 7: Definitions of statutory terms including protective hairstyles historically associated with race (CROWN Act, St. 2022, c.117)
- MGL c. 12, § 11H(b): Right to bias-free professional policing
- MGL c. 93, § 102 & § 103: Massachusetts Equal Rights Act (MERA)
- MGL c. 149, §§ 105A-105C: Discriminatory wage rates based on sex
- MGL c. 151B: Unlawful discrimination because of race, color, religious creed, national origin, ancestry or sex (includes disability discrimination)
- MGL c. 151B, § 4(1E): Pregnant Workers Fairness Act
- MGL c. 272, § 92A: Discrimination in places of public accommodation
- MGL c. 272, § 98: Discrimination in admission to or treatment in place of public accommodation

## Massachusetts Regulations
- 804 CMR: Massachusetts Commission Against Discrimination

## Housing Discrimination Resources
- The Attorney General's guide to landlord and tenant rights: Discrimination in housing is against the law
- Fair housing law, Mass. Attorney General: Includes guidance on preventing housing discrimination based on source of income

## Filing Complaints

Massachusetts agencies:
- Massachusetts Commission Against Discrimination (MCAD): One Ashburton Place, Suite 601, Boston, MA 02108. Main: (617) 994-6000. TTY: (617) 994-6196. MCAD is the state agency responsible for enforcing and investigating violations of MGL 151B, including discrimination in housing.
- Massachusetts Attorney General's Civil Rights Division: Main: (617) 963-2917. Responds to complaints related to civil rights and civil liberties.

Federal agencies:
- U.S. Department of Housing and Urban Development: Report housing discrimination at 202-402-3815
- U.S. Equal Employment Opportunity Commission (Boston): JFK Building, 15 New Sudbury St., Room 475, Boston, MA 02203. Main: (800) 669-4000
""",
))

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151B",
    title="MGL c.151B - Unlawful Discrimination (Table of Contents)",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151B",
    text="""MGL Chapter 151B: UNLAWFUL DISCRIMINATION BECAUSE OF RACE, COLOR, RELIGIOUS CREED, NATIONAL ORIGIN, ANCESTRY OR SEX

Table of Contents:

Section 1: Definitions
Section 2: Policies; recommendations
Section 3: Functions, powers and duties of commission
Section 3A: Employers' policies against sexual harassment; preparation of model policy; education and training programs
Section 4: Unlawful practices
Section 4A: Conveyance by void instruments; penalty
Section 5: Complaints; procedure; limitations; bar to proceeding; award of damages
Section 6: Review of commission's order; court order for enforcement; appeals
Section 7: Posting notices setting forth excerpts of statute and other information
Section 8: Interference with commission; violation of order
Section 9: Construction and enforcement of chapter; civil remedies; attorney's fees and costs; damages
Section 10: Partial invalidity
""",
))

pages.append(dict(
    url="https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151B/Section4",
    title="MGL c.151B s.4 - Unlawful Practices (Housing Discrimination)",
    content_type="statute",
    crawl_depth=1,
    parent_url="https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151B",
    text="""MGL Chapter 151B, Section 4: Unlawful practices

Section 4. It shall be an unlawful practice:

[Subsections 1-5 cover employment discrimination]

## Housing Discrimination Provisions

6. For the owner, lessee, sublessee, licensed real estate broker, assignee or managing agent of publicly assisted or multiple dwelling or contiguously located housing accommodations or other covered housing accommodations to refuse to rent or lease or sell or negotiate for sale or otherwise deny to or withhold from any person or group of persons such accommodations because of the race, color, religious creed, national origin, sex, gender identity, sexual orientation, age, children, ancestry, marital status, veteran status or membership in the armed forces of the United States, status as a person who requires a guide dog, hearing dog, service dog, or other assistive animal, or disability of such person or group or persons or because the person is a recipient of federal, state, or local public assistance, including medical assistance, or because the person is a recipient of federal, state, or local housing subsidies, including rental assistance or rental supplements, or to discriminate against any person because of his race, color, religious creed, national origin, sex, gender identity, sexual orientation, age, children, ancestry, marital status, veteran status or membership in the armed forces of the United States, status as a person who requires a guide dog, hearing dog, service dog, or other assistive animal, or disability in the terms, conditions or privileges of such accommodations or the acquisition thereof, or to make any inquiry or record concerning any of the above.

For purposes of this subsection, discrimination on the basis of handicap includes, but is not limited to, the refusal to allow a handicapped person to make, at his own expense, reasonable modifications to an existing dwelling, and the refusal to make reasonable accommodation for handicapped persons.

7. For the owner, lessee, sublessee, real estate broker, assignee or managing agent of other covered housing accommodations or person having the right of ownership or possession or right to rent or lease, to refuse to rent or lease or sell or negotiate for sale, or otherwise to deny to or withhold from any person or group of persons such accommodations because of race, color, religious creed, national origin, sex, gender identity, sexual orientation, age, ancestry, children, marital status, veteran status or membership in the armed forces of the United States, status as a person who requires a guide dog, hearing dog, service dog, or other assistive animal, disability, or because the person is a recipient of federal, state, or local public assistance, including medical assistance, or because the person is a recipient of federal, state, or local housing subsidies, including rental assistance or rental supplements.

7A. For purposes of subsections 6 and 7, discrimination on the basis of handicap shall include but not be limited to:
(1) a refusal to permit or to make, at the expense of the handicapped person, reasonable modifications of existing premises;
(2) a refusal to make reasonable accommodations in rules, policies, practices, or services, when such accommodations may be necessary to afford such person equal opportunity to use and enjoy a dwelling;
(3) discrimination against or a refusal to rent to a person because of such person's need for reasonable modification or reasonable accommodation.

7B. For any person to make, print, or publish, or cause to be made, printed, or published any notice, statement, or advertisement, with respect to the sale or rental of a dwelling that indicates any preference, limitation, or discrimination based on race, color, religious creed, national origin, sex, gender identity, sexual orientation, age, ancestry, children, marital status, veteran status or membership in the armed forces, disability, or because the person is a recipient of public assistance or housing subsidies.

8. For the owner, lessee, sublessee, or managing agent of, or other person having the right of ownership or possession of commercial space to refuse to rent or lease or to discriminate in terms, conditions or privileges of such commercial space because of the race, color, religious creed, national origin, sex, gender identity, sexual orientation, age, ancestry, or disability of any person.

11. For the owner, sublessees, real estate broker, assignee or managing agent of publicly assisted or multiple dwelling housing accommodations to discriminate against or to refuse to rent to families with children.

Exceptions:
(1) Dwellings containing three apartments or less, one of which apartments is occupied by an elderly or infirm person for whom the presence of children would constitute a hardship.
(2) The temporary leasing or subleasing of a single family dwelling, single apartment, or housing unit for a period not exceeding three calendar months.
(3) The leasing of a single dwelling unit in a two family dwelling, the other occupancy unit of which is occupied by the owner.

13. For any person to directly or indirectly induce, attempt to induce, prevent, or attempt to prevent the sale, purchase, or rental of any dwelling or housing accommodation by:
(a) implicit or explicit representations regarding the entry or prospective entry into the neighborhood of a person of a particular race, color, religion, national origin, sex, gender identity, sexual orientation, children, ancestry, marital status, disability, veteran status, or public assistance recipient status;
(b) unrequested contact for the purpose of inducing to sell or rent;
(c) implicit or explicit false representations regarding the availability of suitable housing;
(d) false representations regarding the listing, prospective listing, sale, or prospective sale of any dwelling.

18. For the owner, lessee, sublessee, licensed real estate broker, assignee, or managing agent of publicly assisted or multiple dwelling or contiguously located housing accommodations or other covered housing to make any distinction, discrimination or restriction on account of race, color, sex, gender identity, sexual orientation, national origin, ancestry, children, marital status, disability, religion, or veteran status.
""",
))

# ─── Batch 4: Rental Assistance Programs ──────────────────────────────────────

pages.append(dict(
    url="https://www.mass.gov/how-to/apply-for-raft-emergency-help-for-housing-costs",
    title="RAFT - Residential Assistance for Families in Transition",
    content_type="guide",
    crawl_depth=1,
    parent_url="https://www.mass.gov/how-to/apply-for-raft-emergency-help-for-housing-costs",
    text="""Apply for RAFT (Emergency Help for Housing Costs)

The Residential Assistance for Families in Transition (RAFT) program provides short-term emergency funding to help you with eviction, foreclosure, loss of utilities, and other housing emergencies.

## What is RAFT?

RAFT provides up to $7,000 per 12-month period so your family can stay in your current home or move to a new one. You may use the money for rent, utilities, moving costs, and mortgage payments.

## Eligibility

You may be eligible for RAFT if:
- You're at risk of homelessness or losing your housing (for example, you received a Notice to Quit or an eviction notice; you're behind on your mortgage; you received a utility shutoff notice; or you can't stay in your home due to health, safety, or other reasons)
- Your income is less than 50% of your city/town's Area Median Income (AMI)
- Your income is less than 60% of your city/town's AMI AND you are at risk of domestic violence

## What You Need to Apply

Gather these documents before you apply:
- ID for Head of Household (such as a state issued driver's license, birth certificate, or passport)
- Proof of Current Housing (such as a lease, tenancy agreement, or tenancy at will agreement)
- Verification of Housing Crisis (such as a Notice to Quit, proof that you are behind on your mortgage, an eviction notice, a utility shutoff notice)
- Income Verification (may be verified automatically)

Your landlord will need to complete a landlord application. After you submit your application, let your landlord or property manager know as soon as possible. They will also need to submit an application within 21 days, or your application will time out.

## How to Apply

You can apply or check an application's status through the Housing Help Hub online. The RAFT application takes 20-30 minutes to complete.

If you can't apply online, contact your local Regional Administering Agency (RAA).

## Application Review

It usually takes fewer than 30 days to get a response after you submit a RAFT application. If your application is denied, you'll get an email about the decision with steps for asking the RAA to review their decision.

Payments usually go out within 14 business days of your application being approved.

## Contact

- RAFT Portal & Regional Administering Agencies (RAAs): Apply or check your application status online
- Massachusetts 2-1-1: Call 211 or (877) 211-6277 (24/7, free, confidential, multilingual)

## Mediation

Massachusetts Community Mediation Centers offer free pre-court mediation between landlords and tenants for lease disputes and eviction cases. Mediation is confidential, voluntary, and non-judgmental.
""",
))

pages.append(dict(
    url="https://www.mass.gov/how-to/review-eligibility-apply-for-emergency-assistance-ea-family-shelter",
    title="Emergency Assistance (EA) Family Shelter - Eligibility & Application",
    content_type="guide",
    crawl_depth=1,
    parent_url="https://www.mass.gov/how-to/review-eligibility-apply-for-emergency-assistance-ea-family-shelter",
    text="""Review Eligibility & Apply for Emergency Assistance (EA) Family Shelter

Learn how to apply for Emergency Assistance (EA) Family Shelter if you are pregnant or have children under 21 years old.

## Eligibility

You can apply if:
- You're a resident of Massachusetts
- Your family's gross income is at or below 115% of the Federal Poverty Guidelines (FPG)
- You're pregnant or have a child under 21

And the reason you need shelter is one of the following:
- No-fault fire, flood, natural disaster, condemnation, or foreclosure
- At risk of domestic violence
- No-fault eviction
- Your children are exposed to a substantial health and safety risk

A family may include parents or guardians, spouses, siblings, stepparents, stepsiblings, or half-siblings.

## Income Requirements (115% FPG)

| Family Size | Max Gross Monthly Income |
|-------------|-------------------------|
| 1 | $1,530 |
| 2 | $2,074 |
| 3 | $2,618 |
| 4 | $3,163 |
| 5 | $3,707 |
| 6 | $4,251 |
| 7 | $4,796 |
| 8 | $5,340 |
| Per Additional Person | +$544 |

## Required Documents

- Proof of identity (driver's license, birth certificate, or passport)
- Proof of family relationship (birth certificate, custody paperwork)
- MA Residency Documents (MassHealth registration, voter/school registration, MA ID)
- Citizenship or Immigration Documents (passport, green card, etc.)
- Documents for Cause of Homelessness (eviction paperwork, etc.)
- Financial Information - Assets & Income (pay stubs, bank statements)
- Consent to a Criminal Background Check (CORI) for all family members 18+

## How to Apply

Online: Visit the Massachusetts Housing Help Hub to register and start an application.
By phone: Call (866) 584-0653, Monday-Friday 8am-5pm.
In person: Visit an HLC office (Boston, Brockton, Chelsea, Greenfield, Hyannis, Lawrence, Lowell, New Bedford, Salem, Springfield, Worcester).

## After Applying

Families may be placed on the EA Family Shelter Contact List when shelter space is limited. Eligible families may also qualify for HomeBASE, which can help pay for first/last month's rent, security deposit, broker's fee, monthly rent payments for up to three years, furniture, moving expenses, and utilities.
""",
))

pages.append(dict(
    url="https://www.mass.gov/info-details/what-is-emergency-assistance-ea-family-shelter",
    title="What is Emergency Assistance (EA) Family Shelter?",
    content_type="guide",
    crawl_depth=1,
    parent_url="https://www.mass.gov/info-details/what-is-emergency-assistance-ea-family-shelter",
    text="""What is Emergency Assistance (EA) Family Shelter?

Shelter is a temporary place for your family to stay as you search for your next place to live. You may be eligible for EA Family Shelter if you have lost your home due to unforeseen circumstances such as natural disasters, fleeing domestic violence, foreclosures, and other criteria.

## What is Expected from Families in EA Family Shelter?

While you stay in shelter there will be required tasks:
- Upload Documentation: Hand in all remaining required documents
- Do Re-Housing Activities: Family members 18 and over will take steps to stabilize and find housing
- Attend Meetings: Attend shelter meetings and workshops in your Re-Housing Plan
- Accept Permanent Housing: Accept an offer of permanent housing unless you have good cause
- Follow Shelter Program Rules: Arriving by curfew, spending every night at shelter, providing appropriate care for children, no pets (except service animals), no alcohol or illegal drug use

## Types of Shelters

- Congregate: Families have their own private place to sleep but share common spaces with other families.
- Co-Shelter: Families are placed in apartments with other families, with private bedrooms and shared common spaces.
- Scattered Site: Families are placed in private living spaces/apartments separate from other families.

## Tracks of EA Family Shelter

As of December 15, 2025, all families entering the EA Family Shelter Program are eligible for Bridge Shelter.

Bridge Shelter Track: Families work to find stable, permanent housing quickly and are assisted with employment and health resources. Families can stay for up to six months, with possible short extensions.

Rapid Shelter Track (used if program runs out of space): Families can stay for up to 30 business days, with possible short extensions.

## Key Terms

- Case Worker (Family Advocate/Case Manager): Person at the shelter who assists you in planning how to stabilize your living situation.
- Diversion Provider: Organization that supports eligible families to pay costs to travel to a safe place or find housing instead of entering shelter.
- HomeBASE: A program that can help you pay for part of the rent for an apartment or find alternative housing instead of going to a shelter.
- Homeless Coordinator: The person who helps you complete your application and determines eligibility.
- Housing Search Worker: Person who assists you specifically in finding housing.

## Contact

Massachusetts Emergency Family Shelter Contact Line: (866) 584-0653, Monday-Friday 8am-5pm.
""",
))

# ─── Save all pages ───────────────────────────────────────────────────────────

def main():
    saved = []
    failed = []

    for i, page in enumerate(pages, 1):
        print(f"\n[{i}/{len(pages)}] Saving: {page['title'][:60]}...")
        try:
            result = save_browser_page(
                url=page["url"],
                title=page["title"],
                text=page["text"],
                content_type=page["content_type"],
                crawl_depth=page["crawl_depth"],
                parent_url=page.get("parent_url"),
            )
            if result:
                saved.append(result)
                print(f"  ✓ Saved: {result['doc_id']}")
            else:
                failed.append(page["title"])
                print(f"  ✗ Skipped (too short?)")
        except Exception as e:
            failed.append(page["title"])
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"Saved: {len(saved)} / {len(pages)} pages")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
