"""
Utility Functions for Indian Real Estate Context.
Includes INR formatting, Vastu scoring, Festive modifiers, EMI & Stamp Duty math.
"""

import numpy as np

def format_inr(amount):
    """Formats numeric amounts into traditional Indian currency style (Lakhs and Crores)."""
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return f"₹ {amount}"
    
    # Handle sign
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    
    if amount >= 10000000: # 1 Crore = 10,000,000 INR
        return f"{sign}₹ {amount / 10000000:.2f} Cr"
    elif amount >= 100000: # 1 Lakh = 100,000 INR
        return f"{sign}₹ {amount / 100000:.2f} Lakh"
    else:
        return f"{sign}₹ {amount:,.2f}"

def vastu_score(facing, vastu_compliant_status):
    """
    Calculates a Vastu Score (0-100) based on property facing direction and general Vastu status.
    In Vastu Shastra:
    - East and North-East facings are highly positive.
    - South and South-West are generally considered less auspicious and need corrections.
    """
    # 1. Base Score based on Facing Direction (Max 50)
    facing_map = {
        'North-East': 50,
        'East': 45,
        'North': 42,
        'North-West': 32,
        'West': 28,
        'South-East': 20,
        'South': 15,
        'South-West': 10
    }
    base_facing = facing_map.get(facing, 25)
    
    # 2. Compliance Score (Max 50)
    compliance_map = {
        'Yes': 50,
        'Partially': 30,
        'No': 10
    }
    base_compliance = compliance_map.get(vastu_compliant_status, 20)
    
    score = base_facing + base_compliance
    return min(100, max(0, score))

def get_vastu_significance(facing):
    """Returns Vastu guidance descriptions for each facing direction."""
    significance = {
        'East': "Highly auspicious. Governed by Sun. Associated with growth, health, and positive energy flow.",
        'North-East': "Most auspicious direction (Ishan corner). Governed by Shiva/Water. Brings peace, intelligence, and prosperity.",
        'North': "Excellent. Governed by Kubera (God of wealth). Promotes business opportunities and monetary gains.",
        'North-West': "Neutral. Governed by Vayu (Wind). Relates to travel, relationships, and business partnerships.",
        'West': "Neutral-Positive. Governed by Varuna. Associated with stability, discipline, and steady career growth.",
        'South-East': "Challenging (Agni corner). Governed by Fire. Suitable for kitchens; otherwise requires remedies to avoid health issues.",
        'South': "Challenging. Governed by Yama (God of death). Can be stabilized if the main entrance has heavy gates/remedies.",
        'South-West': "Highly challenging (Nairutya corner). Governed by Rahu. Demands heavy construction, thick walls to block negative elements."
    }
    return significance.get(facing, "Standard Vastu properties apply.")

def festive_multiplier(month_num):
    """
    Returns price multiplier based on India's major festive calendars and purchase demand cycles:
    - Diwali (Oct-Nov): +7% to +10% premium due to massive auspicious buying waves and bonuses.
    - Navratri / Ugadi / Gudi Padwa / Pongal (Jan, Mar, Apr): +3% to +5% premium.
    - Monsoon (Jul-Aug): -3% to -5% discount due to construction halts and wet site visits.
    - Year-End Clearance (Dec): -6% discount.
    """
    if month_num in [10, 11]:  # October, November (Diwali / Durga Puja)
        return 1.08, "Festive Season (Diwali / Durga Puja Peak Purchase Demand) (+8%)"
    elif month_num in [1, 3, 4]:  # January (Pongal/Makar Sankranti), March/April (Navratri/Gudi Padwa/Ugadi)
        return 1.04, "Auspicious Buying Cycle (Navratri / Pongal / Ugadi) (+4%)"
    elif month_num in [7, 8]:  # July, August (Monsoons)
        return 0.97, "Monsoon Slump Discount (-3%)"
    elif month_num == 12:  # December (Year-End Clearance)
        return 0.94, "Year-End Developer Clearance Discount (-6%)"
    else:
        return 1.00, "Standard Market Rate (No Seasonal Fluctuations)"

def emi_calculation(principal_inr, rate_percent, tenure_years):
    """Calculates Monthly EMI, total payment, and interest payable for Home Loans."""
    P = float(principal_inr)
    r = (float(rate_percent) / 100) / 12  # Monthly interest rate
    n = int(tenure_years * 12)  # Monthly periods
    
    if r == 0:
        emi = P / n
        total_payment = P
        total_interest = 0
    else:
        # Standard EMI Formula
        emi = P * r * ((1 + r)**n) / (((1 + r)**n) - 1)
        total_payment = emi * n
        total_interest = total_payment - P
        
    return emi, total_payment, total_interest

def stamp_duty_calculation(city, price_inr):
    """
    Calculates Stamp Duty based on state regulations in Indian cities:
    - Mumbai / Pune (Maharashtra): ~6%
    - Delhi NCR (Delhi): ~6%
    - Bangalore (Karnataka): ~5%
    - Hyderabad (Telangana): ~6%
    - Chennai (Tamil Nadu): ~7%
    - Ahmedabad (Gujarat): ~4.9%
    - Jaipur (Rajasthan): ~6%
    - Lucknow (Uttar Pradesh): ~7%
    - Indore (Madhya Pradesh): ~7.5%
    """
    city_rates = {
        'Mumbai': 0.06,
        'Delhi NCR': 0.06,
        'Bangalore': 0.05,
        'Hyderabad': 0.06,
        'Pune': 0.06,
        'Chennai': 0.07,
        'Ahmedabad': 0.049,
        'Jaipur': 0.06,
        'Lucknow': 0.07,
        'Indore': 0.075
    }
    rate = city_rates.get(city, 0.06)
    stamp_duty = price_inr * rate
    
    # Registration charges are typically 1% of property value (capped at 30k in some states,
    # but let's assume a standard 1% registration fee up to a max cap of 30,000 INR)
    registration_fee = min(30000.0, price_inr * 0.01)
    
    return stamp_duty, registration_fee, rate
