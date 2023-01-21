from __future__ import division
from pyutilib.misc.timing import tic, toc
import pyomo.environ as en
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import collections
pd.options.mode.chained_assignment = None  # default='warn'

# %%  model formulation
# This version of model includes separate power and energy costs for battery to endogenize duration selection
###################################### REVISIT THIS FUNCTION DEFINTION###############
def build_model(pvavailarray,pricearray,dfPVData,dfStorData,dfElyData,dfH2StData,CCF,
                productionCommitmentLB=None,minimumProductionShutdownLength=0):
    # pvavailarray - hourly PV capacity factor for location
    # pricearray - hourly grid price
    # dfPVData - cost information of PV and inverter
    # dfStorData - cost information for Storage 
    # dfElyData - cost information for electrolyzer
    # dfH2StData - cost information for H2 storage
    # CCF- capital charge rate to convert CAPEX to annualized CAPEX
    # requiredUptime - # of hours required in production bands
    if productionCommitmentLB is None:
        productionCommitmentLB = len(pvavailarray)

    tic()
    m = en.ConcreteModel()

    tval =range(1,len(pvavailarray)+1,1)

    esval = dfStorData.index.values.tolist()

# Time periods of optimization
    m.t = en.Set(initialize =tval)

#Set of battery storage technologies
    m.bes = en.Set(initialize = esval)
    
#    # Set of H2 storage technologies
#    m.h2es = en.Set(initialize =h2esval)
    # Capital charge factor
    m.pCCF = en.Param(within= en.NonNegativeReals, initialize = CCF)

# Parameters --------------------------------------------------
    #H2 production rate in kg/hr
    m.pH2DesignFlowRate = en.Param(within =en.NonNegativeReals, initialize = 4166.667)
    
    # lower bound on plant utilization factor
    m.pCFPlantLB = en. Param(within=en.NonNegativeReals, initialize = 0.9) 
    
    # H2 LHV in MJ/kg
    m.pH2LHV = en.Param(within=en.NonNegativeReals, initialize =120.1)

# Cost of unmet of H2 demand - $/kg - default value= 0 as we are not modeling slacks
    #Slacks only become important when modeling dispatch of grid to evaluate impact of exports
    m.pProductionSlackCost = en.Param(within=en.NonNegativeReals, initialize=0, mutable=True)
    
## PV parameters-------------------------------------------------------------
    # PV availability - capacity factor
    m.pPVCapFactor = en.Param(m.t,within=en.NonNegativeReals, 
                              initialize = {tval[j]:pvavailarray[j] for j in range(len(tval))}, mutable=True)
    #capital cost of PV - mutable - $/kW without inverter cost
    m.pCapCostPV = en.Param(within=en.NonNegativeReals,
                            initialize= dfPVData.CapCost_dkW.values[0])
    
    # Fixed O&M cost of PV system $/MW/yr - calculated as percent of capex
    m.pFOMCostPV = en.Param(within=en.NonNegativeReals,
                            initialize=dfPVData.FOM_pct_CAPEX.values[0]*dfPVData.CapCost_dkW.values[0]*1000)
    
    #Variable O&M cost of PV system $/MWh
    m.pVOMCostPV = en.Param(within=en.NonNegativeReals,
                            initialize =dfPVData.VOM_dMWh.values[0])
    
    #Grid electricity wholesale price in $/MWh
    m.pGridElecPrice = en.Param(m.t, initialize={tval[j]:pricearray[j] for j in range(len(tval))}, mutable=True)
        

# Inverter parameters --------------------------------------------------------
# Inverter DC to AC efficiency
    m.pInvEff = en.Param(within=en.NonNegativeReals, initialize = dfPVData.InvEff.values[0])
    
    #Capital cost of inverter in $/kW
    m.pCapCostInv = en.Param(within=en.NonNegativeReals, initialize = dfPVData.InvCapCost_dkW.values[0])

# Battery storage parameters -------------------------------------------------
# Efficiency of charging
    m.pStEffChg = en.Param(m.bes, within=en.NonNegativeReals, 
                           initialize = {esval[j]:dfStorData.loc[esval[j],'St_eff_chg'] for j in range(len(esval))})

# Efficiency of discharging
    m.pStEffDischg = en.Param(m.bes, within=en.NonNegativeReals, 
                              initialize ={esval[j]:dfStorData.loc[esval[j],'St_eff_dischg'] for j in range(len(esval))})
      
#  capital cost of storage - power cost - $/kW
    m.pCapCostPowSt = en.Param(m.bes, within=en.NonNegativeReals, 
                            initialize ={esval[j]:dfStorData.loc[esval[j],'Power_capex_dpkW'] for j in range(len(esval))})

#  capital cost of storage - energy cost - $/kWh
    m.pCapCostEnergySt = en.Param(m.bes, within=en.NonNegativeReals, 
                            initialize ={esval[j]:dfStorData.loc[esval[j],'Energy_capex_dpkWh'] for j in range(len(esval))})

# Fixed O&M costs - $/MW/yr
    m.pFOMCostSt = en.Param(m.bes, within=en.NonNegativeReals, 
                            initialize ={esval[j]:dfStorData.loc[esval[j],'FOM_dMWyr'] for j in range(len(esval))})

# Variable O&M costs- $/MWh
    m.pVOMCostSt = en.Param(m.bes, within=en.NonNegativeReals,
                            initialize ={esval[j]:dfStorData.loc[esval[j],'VOM_dMWh'] for j in range(len(esval))})

# Upper bound on storage duration - hours
    m.pDur_UB = en.Param(m.bes,within=en.NonNegativeReals,
                         initialize={esval[j]:dfStorData.loc[esval[j],'MaxDur_hrs'] for j in range(len(esval))})
# Electrolyzer parameters -------------------------------------------------   
    # Electrolyzer efficiency (kWh/kg H2)
    m.pElySpecPower =en.Param(within =en.NonNegativeReals, initialize =dfElyData.ElySpecPower_kWhkg.values[0] )
    
    #Capex of Electrolysis plant  $/kW of electricity input
    m.pCapCostEly = en.Param(within =en.NonNegativeReals, initialize = dfElyData.CapCost_dkW.values[0])
    
    # FOM O&M cost of electrolyzer $/MW-yr - calculated as a percent of capital costs
    m.pFOMCostEly = en.Param(within =en.NonNegativeReals,
                             initialize = dfElyData.FOM_pct_CAPEX.values[0]*dfElyData.CapCost_dkW.values[0]*1000)
    
    # Variable O&M cost of electrolyzer $/MWh  of electricity input
    m.pVOMCostEly = en.Param(within =en.NonNegativeReals,initialize = dfElyData.VOMCost_dMWh.values[0])

    # Feed water cost $/kg H2 produced
    m.pFeedH2OCostEly = en.Param(within =en.NonNegativeReals,initialize = dfElyData.Water_cost_d_per_kg_H2.values[0])
    
# H2 storage and compressor parameters
    # Compressor capital costs in $/kW
    m.pCapCostH2Comp = en.Param(within=en.NonNegativeReals, initialize =dfH2StData.CapCostComp_dkW.values[0])

    # FOM O&M cost of compressor $/MW-yr - calculated as a percent of capital costs
    m.pFOMCostH2Comp = en.Param(within =en.NonNegativeReals,
                             initialize = dfH2StData.CompFOM_pct_CAPEX.values[0]*dfH2StData.CapCostComp_dkW.values[0]*1000)

    
    #Compressor specific power to go from 30 bar to 350 bar storage pressure - kWh/kg
    m.pCompSpecPower = en.Param(within=en.NonNegativeReals, initialize = dfH2StData.CompSpecPower_kWhpkg.values[0])

    # Storage capital costs in $/kg h2 stored
    m.pCapCostH2st = en.Param(within=en.NonNegativeReals,initialize =dfH2StData.CapCostst_dkg.values[0])    
    
    # FOM O&M cost of H2 storage $/kg/yr - calculated as a percent of capital costs
    m.pFOMCostH2st = en.Param(within =en.NonNegativeReals,
                             initialize = dfH2StData.StFOM_pct_CAPEX.values[0]*dfH2StData.CapCostst_dkg.values[0])
    
    # Mass of H2 stored per tank kg
    m.pH2kgpertank = en.Param(within=en.NonNegativeReals, initialize =dfH2StData.mass_stored_kg.values[0])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------
# Variables ----------------------------------------
# PV installed capacity (DC)
    m.vPVInstalledMW = en.Var(within =en.NonNegativeReals)
    
# PV power output to power gen block MW
    m.vPVtoDCPow = en.Var(m.t, within = en.NonNegativeReals)
    
# Total Power generated from PV - excluding curtailment
    m.vPVOutput = en.Var(m.t, within=en.NonNegativeReals)

# Inverter installed capacity in MW
    m. vInvInstalledMW = en.Var(within=en.NonNegativeReals)
      
# Storage related variables ---------------------------------------------------
#Storage discharge to power generation block-MW
    m.vStDischarge = en.Var(m.t, m.bes, within=en.NonNegativeReals)

#Storage charge from PV system - MW
    m.vStCharge = en.Var(m.t, m.bes, within=en.NonNegativeReals)

#Storage state of charge -MWh
    m.vStSoC = en.Var(m.t, m.bes, within=en.NonNegativeReals)

#Installed storage power capacity -- VAR -- MW
    m.vStInstalledMW = en.Var(m.bes, within = en.NonNegativeReals)

#Installed storage energy capacity -- VAR -- MWh
    m.vStInstalledMWh = en.Var(m.bes, within = en.NonNegativeReals)

# H2 electrolyzer variables --------------------------------------------------
           
    #Installed capacity of electrolyzers MW
    m.vElyInstalledMW = en.Var(within=en.NonNegativeReals)

    # power sent to electrolyzer in MW
    m.vDCPowtoEly = en.Var(m.t,within =en.NonNegativeReals)


# H2 compressor,storage and inverter variables --------------------------------
    
    #Comp capacity rating MW  - also identical to inverter capcity rating
    m.vCompInstalledMW = en.Var(within=en.NonNegativeReals)
    
    #H2 storage capacity - number of tanks - linearized
    m.vH2StInstalledNumber = en.Var(within=en.NonNegativeReals)

    # H2 flow leaving the plant kg/hr
    m.vH2flowProd = en.Var(m.t, within =en.NonNegativeReals)
    
    # H2 flow to storage unit kg/hr
    m.vH2flowStorage =en.Var(m.t, within=en.NonNegativeReals)
        
    # power sent to inverter in MW
    m.vDCPowtoInv = en.Var(m.t,within =en.NonNegativeReals) 
    
    # power sent to compressor in MW
    m.vACPowtoComp = en.Var(m.t, within=en.NonNegativeReals)
    
    # power sent to compressor in MW
    m.vACPowtoGrid = en.Var(m.t, within=en.NonNegativeReals)
    
        #H2 storage number of tanks
    m.vH2StStatekg = en.Var(m.t, within =en.NonNegativeReals)
    
    #H2 storage discharge in kg/hr
    m.vH2StflowProd = en.Var(m.t, within=en.NonNegativeReals)

# Total H2 production in each hour from storage + electrolyzer
    # H2 plant leaving the process
    m.vTotalH2Output = en.Var(m.t, within =en.NonNegativeReals)

# Absolute difference outside of production bands -kg/hr
    # Set this to zero in case grid interactions are ignored
    m.vH2PlantOutputSlack = en.Var(m.t, within=en.NonNegativeReals, 
                                       initialize={tval[j]:0 for j in range(len(tval))})

# Objective function terms ----------------------------------------------------
    print(f'defined parameters {int(toc())}')
    
# PV fixed costs in $
    def PVFixCost_rule(m):
        return m.vPVInstalledMW*(m.pCCF*m.pCapCostPV*1000) + m.vPVInstalledMW*m.pFOMCostPV
    m.ePVFixCost = en.Expression(rule =PVFixCost_rule)
    print(f'defined PV fixcost {int(toc())}')
    
#  Storage fixed costs in $ - separately evaluate power and energy costs
    def StorFixCost_rule(m):
        return sum(1000*m.pCapCostPowSt[st]*m.vStInstalledMW[st]*m.pCCF +
                   1000*m.pCapCostEnergySt[st]*m.vStInstalledMWh[st]*m.pCCF+
                   m.pFOMCostSt[st]*m.vStInstalledMW[st] for st in m.bes)
    m.eStFixCost = en.Expression(rule =StorFixCost_rule)
    
# Electrolyzer fixed costs in $
    def ElyFixCost_rule(m):
        return m.pCapCostEly*m.vElyInstalledMW*m.pCCF*1000+ m.pFOMCostEly*m.vElyInstalledMW
    m.eElyFixCost = en.Expression(rule =ElyFixCost_rule)
    
# H2 storage, compressor and inverter capital costs - Inverter rating set by amount of grid power exports + compressor power required
    def H2StFixCost_rule(m):
        val= m.vCompInstalledMW*m.pCapCostH2Comp*m.pCCF*1000 + \
        m.vCompInstalledMW*m.pFOMCostH2Comp +\
        m.vInvInstalledMW*m.pCapCostInv*m.pCCF*1000 + \
        m.vH2StInstalledNumber*m.pH2kgpertank*m.pCapCostH2st*m.pCCF +\
        m.vH2StInstalledNumber*m.pH2kgpertank*m.pFOMCostH2st
        return val
    m.eH2StFixCost = en.Expression(rule =H2StFixCost_rule)
    print(f'defined H2 fixcost {int(toc())}')

#System variable operating  costs - PV + storage + Electrolyzer (including water feed cost)
    def SysVOMCost_rule(m):
        val = sum(sum(m.vStCharge[t,st] + m.vStDischarge[t,st] for t in m.t)*m.pVOMCostSt[st] for st in m.bes) + sum(m.vH2flowProd[t]*m.pFeedH2OCostEly + m.vH2flowStorage[t]*m.pFeedH2OCostEly for t in m.t)
#              + m.vH2flowStorage[t]*m.pCompSpecPower*10/1000 # $2/MWh VOM costs for H2 storage charging
####             + m.vPVOutput[t]*m.pVOMCostPV   \
####             + m.vDCPowtoEly[t]*m.pVOMCostEly for t in m.t)
        return val
    m.eSysVOMCost = en.Expression(rule = SysVOMCost_rule)
    print(f'defined sys variable cost {int(toc())}')
##

# Revenue from grid electricity sales in $
    def RevenueGridSale_rule(m):
        val = sum(m.vACPowtoGrid[t]*m.pGridElecPrice[t] for t in m.t)
        return val
    m.eGridRevenue = en.Expression(rule =RevenueGridSale_rule)

     # Slack is being modeled
    def ProductionSlack_rule(m):
        return m.pProductionSlackCost * sum(m.vH2PlantOutputSlack[t] for t in m.t)
    m.eProductionSlack = en.Expression(rule=ProductionSlack_rule)

# Objective function value -minimize total annualize system costs after accounting for co-product (elec) revenue
    def SysTotalCost(m):
        return m.ePVFixCost + m.eStFixCost + m.eElyFixCost + m.eH2StFixCost +m.eSysVOMCost - m.eGridRevenue
    m.eSysTotalCost = en.Expression(rule=SysTotalCost)

# Expression for model objective function 
    def Obj_fn(m):
        return m.eSysTotalCost + m.eProductionSlack
    m.oObjective = en.Objective(rule=Obj_fn, sense =en.minimize)
    print(f'defined objective function {int(toc())}')

# constraints -----------------------------------------------------------

# PV energy balance - sum of storage charge  and DC power sent to Pow block less than PV output (DC)
    def PVEnergyBalance(m, t):
#        return sum(m.vP2Charge[t,st] for st in m.es) + m.vPGrid[t] + m.vExcessGen[t] == m.pPVDC_MW*m.pPVCapFactor[t]
        return sum(m.vStCharge[t, st] for st in m.bes) + m.vPVtoDCPow[t]  == m.vPVOutput[t]
    m.cPVEnergyBal = en.Constraint(m.t, rule = PVEnergyBalance)

# Capacity rating constraint by availability
    def PVCapLim(m,t):
        return m.vPVOutput[t]<=m.vPVInstalledMW*m.pPVCapFactor[t]
    m.cPVCapLim = en.Constraint(m.t, rule = PVCapLim)
    print(f'defined PV constraints {int(toc())}')

# Storage constraints -----------------------------------------------------------
    
# Balance storage capacity at each time step (MWh)
    def StSoCBal(m,t,st): # wrapping storage capacity to ensure first and last period are matching
        if t in [1]:  # First hour of the year
            #  first hour specific constraints- wrapping constraints across the year
            return m.vStSoC[len(pvavailarray),st] +m.pStEffChg[st]*m.vStCharge[t,st] -  \
        m.vStDischarge[t,st]/m.pStEffDischg[st] == m.vStSoC[t,st]
        
        else:
            return m.vStSoC[t-1,st]+ m.pStEffChg[st]*m.vStCharge[t,st] -   \
        m.vStDischarge[t,st]/m.pStEffDischg[st] == m.vStSoC[t,st]
        
    m.cStSoCBal = en.Constraint(m.t, m.bes, rule=StSoCBal)

# Upper limit on charge rate into the battery
    def StLimChargeUB(m, t, st):
        return m.pStEffChg[st]*m.vStCharge[t,st] <= m.vStInstalledMW[st] 
    m.cStChargeUB = en.Constraint(m.t, m.bes, rule = StLimChargeUB)

# Upper limit on discharge rate into the battery
    def StLimDischargeUB(m, t,st):
        return m.vStDischarge[t,st] <=m.pStEffDischg[st]*m.vStInstalledMW[st]
    m.cStLimDischargeUB = en.Constraint(m.t, m.bes, rule = StLimDischargeUB)

# Storage capacity cannot exceed purchased energy capacity
    def StCap_rule(m,t,st):
        return m.vStSoC[t,st]<=m.vStInstalledMWh[st]
    m.cSt_Cap = en.Constraint(m.t, m.bes, rule = StCap_rule)
   

# Storage duration upper bounded by specified parameter value hours
    def StDur_UB(m,st):
        return m.vStInstalledMWh[st]<=m.pDur_UB[st]*m.vStInstalledMW[st]
    m.cStDur_UB = en.Constraint(m.bes, rule = StDur_UB)
    
    print(f'defined storage constraints {int(toc())}')
   

# DC Power splitter block balance -----------------------------------------------------------
    def DCPowerBlockBal(m,t):
        return sum(m.vStDischarge[t,st] for st in m.bes) + m.vPVtoDCPow[t]  \
        == m.vDCPowtoEly[t] + m.vDCPowtoInv[t]
    m.cDCPowerBal = en.Constraint(m.t,rule =DCPowerBlockBal)
    
    
# electrolyzer energy balance --------------------------------------------
    # Multiplying DC power by 1000 to convert to kW - 1 hour resolution unis are kg/hr left and right hand side
    def ElyEnergyBal(m,t):
        return m.vDCPowtoEly[t]*1000/m.pElySpecPower == m.vH2flowProd[t] + m.vH2flowStorage[t]
    m.cElyEnergyBal = en.Constraint(m.t, rule = ElyEnergyBal)

# capacity balance on electrolyzer
    def ElyCapLim(m,t):
        return m.vDCPowtoEly[t]<=m.vElyInstalledMW
    m.cElyCapLim = en.Constraint(m.t, rule = ElyCapLim)
    
# Storage, compressor and inverter balance ----------------------------------    
 
#Inverter energy balance in MW
    def InvEnergyBal(m,t):
        return m.vDCPowtoInv[t]*m.pInvEff==m.vACPowtoComp[t] +m.vACPowtoGrid[t]
    m.cInvEnergyBal = en.Constraint(m.t, rule = InvEnergyBal)

#Inverter capacity rating in MW
    def InvCapLim(m,t):
        return m.vDCPowtoInv[t] <= m.vInvInstalledMW
    m.cInvCapLim = en.Constraint(m.t, rule = InvCapLim)

  
#Compressor capacity rating inMW
    def CompCapLim(m,t):
        return m.vACPowtoComp[t] <= m.vCompInstalledMW
    m.cCompCapLim = en.Constraint(m.t, rule = CompCapLim)
    
#Compressor energy requirement -CompSpecPower unit MJ/kg of H2, units of equation MW
    def CompPowReq(m,t):
        return m.vACPowtoComp[t] == m.vH2flowStorage[t]*m.pCompSpecPower/1000
    m.cCompPowReq = en.Constraint(m.t, rule = CompPowReq)

# H2 storage balance ---------------------------------------------------------

# Balance storage capacity at each time step (kg) - no effiiciency losses assumed
    def H2StSoCBalance(m,t): # wrapping storage capacity to ensure first and last period are matching
        if t in [1]:  # First hour of the yeardefints across the year
            return m.vH2StStatekg[len(pvavailarray)] +m.vH2flowStorage[t] -  \
        m.vH2StflowProd[t] == m.vH2StStatekg[t]
        
        else:
            return m.vH2StStatekg[t-1] +m.vH2flowStorage[t] -  \
        m.vH2StflowProd[t] == m.vH2StStatekg[t]
        
    m.cH2StBal = en.Constraint(m.t, rule=H2StSoCBalance)

# Capacity rating for H2 storage cannot exceed total storage
    def H2StCapLim(m,t):
        return m.vH2StStatekg[t] <=m.vH2StInstalledNumber*m.pH2kgpertank
    m.cH2StCapLim = en.Constraint(m.t, rule =H2StCapLim)
    print(f'defined H2 storage constraints {int(toc())}')

# Plant H2 balance -----------------------------------------------------------    
# Total plant H2 production in kg/hr
    def TotalPlantH2bal(m,t):
        return m.vH2flowProd[t] + m.vH2StflowProd[t]== m.vTotalH2Output[t]
    m.cTotalPlantH2bal =en.Constraint(m.t,rule = TotalPlantH2bal)
    
    # Binary variable for each time period
    m.vProductionCommitment = en.Var(m.t, within=en.Binary)

    #m.pProductionCommitmentLB = en.Param(within=en.NonNegativeIntegers, initialize=productionCommitmentLB, mutable=True)

    # Lower bound on the # of hours shutdown
    def ProductionCommitmentLB(m):
        return sum(m.vProductionCommitment[t] for t in m.t) >= productionCommitmentLB#m.pProductionCommitmentLB
    m.cProductionCommitmentLB = en.Constraint(rule=ProductionCommitmentLB)

    #m.pMinimumProductionShutdownLength = en.Param(within=en.NonNegativeIntegers, initialize=minimumProductionShutdownLength, mutable=True)

    # For every startup, require the last T periods to be shutdown
    """
    def ProductionCommitmentContiguity(m,t):
        # cyclic
        return en.quicksum([1-m.vProductionCommitment[((t-i-1)%len(pvavailarray)) + 1] for i in range(1, minimumProductionShutdownLength+1)]) >= \
               minimumProductionShutdownLength * (m.vProductionCommitment[t] - m.vProductionCommitment[(t-2)%len(pvavailarray) + 1]) # startup = 1
        # non-cyclic
        #if t == 1:
        #    return en.Constraint.Skip
        #else:
        #    return en.quicksum([1-m.vProductionCommitment[i] for i in range(max(1, t-minimumProductionShutdownLength-1), t)]) >= \
        #           minimumProductionShutdownLength * (m.vProductionCommitment[t] - m.vProductionCommitment[t-1]) # startup = 1
    m.cProductionCommitmentContiguity = en.Constraint(m.t, rule=ProductionCommitmentContiguity)

    """
    def ProductionCommitmentContiguity(m,t,offset):
        def ind(i):
            return ((i-1) % len(pvavailarray)) + 1
        k = t - offset
        return 1 - m.vProductionCommitment[ind(k)] >= m.vProductionCommitment[ind(t)] - m.vProductionCommitment[ind(t-1)]

    m.sContiguityOffset = en.RangeSet(1, minimumProductionShutdownLength)
    m.cProductionCommitmentContiguity = en.Constraint(m.t, m.sContiguityOffset, rule=ProductionCommitmentContiguity)

# H2 plant output generation requirements 
    def H2PlantOutputLB_rule(m,t):
        if en.value(m.pProductionSlackCost) ==0: #- in the absence of slacks
            return m.vTotalH2Output[t] >= m.pH2DesignFlowRate*m.pCFPlantLB*m.vProductionCommitment[t]
        else:# H2 plant output generation requirements with slacks
            return -m.vH2PlantOutputSlack[t] <= m.vTotalH2Output[t] - m.pH2DesignFlowRate*m.pCFPlantLB*m.vProductionCommitment[t]
    m.cH2PlantOutputLB = en.Constraint(m.t, rule= H2PlantOutputLB_rule)

    def H2PlantOutputUB_rule(m,t):
        if en.value(m.pProductionSlackCost) ==0: #- in the absence of slacks
            return m.vTotalH2Output[t] <= m.pH2DesignFlowRate*m.vProductionCommitment[t]
        else: # H2 plant output generation requirements with slacks
            return m.vH2PlantOutputSlack[t] >= m.vTotalH2Output[t] - m.pH2DesignFlowRate*m.vProductionCommitment[t]         
    m.cH2PlantOutputUB = en.Constraint(m.t, rule= H2PlantOutputUB_rule)
   
    
        
    return m

def GetStaticOutputs(instancename):
    # Storing all the outputs in a single dictionary
#    OutputsbyUnitbyVarName =collections.OrderedDict()
    
    Timesteps = np.arange(1,len(instancename.t.data())+1)
    StorUnits = len(instancename.bes.data())
    
    StorIndex= [f'es{i}' for i in range(1,StorUnits+1)]
    
    #  Input parameters
#    InputPars = ['PV_CAPEX_dpkW','Ely_CAPEX_dpkW','H2St_CAPEX_dpkg','Comp_CAPEX_dpkW','PV_CF']
    StorInputParsPow =[StorIndex[i] +'_CAPEX_dpkW' for i in range(len(StorIndex))]
    StorInputParsEnergy =[StorIndex[i] +'_CAPEX_dpkWh' for i in range(len(StorIndex))]
    StorPowerOutputs = [StorIndex[i] +'_InstalledCapMW' for i in range(len(StorIndex))]
    StorEnergyOutputs = [StorIndex[i] +'_InstalledCapMWh' for i in range(len(StorIndex))]
    #    DesignOutputs = ['PV_DC_MW','Ely_MW','H2St_tonne']
#    SystemMetrics = ['H2ProdCost_perkg', 'H2ProdCost_perGJ','PV_curtailment','ElyCF','ElyToPVCap']
    
#    StaticOutputsCols = InputPars + StorInputPars+ DesignOutputs + StorDesignOutputs+ SystemMetrics
    
#    df1 = pd.DataFrame(np.zeros([1,len(StaticOutputsCols)]),columns =StaticOutputsCols)
    df1 = collections.OrderedDict()
    
    df1['PV_CAPEX_dpkW'] =  en.value(instancename.pCapCostPV) # Capex of PV ($/kW)
    df1['Ely_CAPEX_dpkW'] =  en.value(instancename.pCapCostEly)  # Capex of Electrolyzer ($/kW)
    df1['H2St_CAPEX_dpkg'] =  en.value(instancename.pCapCostH2st) # Capex of H2 storage ($/kg)
        
    df1['Comp_CAPEX_dpkW'] =  en.value(instancename.pCapCostH2Comp) # Capex of H2 compressor ($/kW)
    df1['Ely_FOM_pct_CAPEX'] =  en.value(instancename.pFOMCostEly)/en.value(instancename.pCapCostEly*1000) # FOM cost as percent of CAPEX
    df1['PV_CF'] =  np.mean([en.value(instancename.pPVCapFactor[t]) for t in Timesteps]) # Average capacity factor of PV availability
    df1['CCF'] =  en.value(instancename.pCCF) # Capital charge fraction
    df1['H2Designkghr'] =  en.value(instancename.pH2DesignFlowRate) # Design flow rate in kg/hr
    df1['AvgH2FlowRate_kghr']= sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps])/len(Timesteps) # Average production rate kg/hr
    df1['Ely_SpecPower'] =  en.value(instancename.pElySpecPower) # Specific power consumption kWh/kg


    
    for i in range(len(StorIndex)):    # storage related input capital costs
        df1[StorInputParsPow[i]] =  en.value(instancename.pCapCostPowSt[StorIndex[i]]) # energy Capex of each storage technology($/kW)
        df1[StorInputParsEnergy[i]] =  en.value(instancename.pCapCostEnergySt[StorIndex[i]]) # energy Capex of each storage technology($/kWh)
        df1[StorPowerOutputs[i]] =  en.value(instancename.vStInstalledMW[StorIndex[i]]) # Installed Storage power (MW)
        df1[StorEnergyOutputs[i]] =  en.value(instancename.vStInstalledMWh[StorIndex[i]]) # Installed Storage capacity (MWh)

    df1['PV_DC_MW'] =  en.value(instancename.vPVInstalledMW) # PV DC capacity (MW)
    df1['Ely_MW'] =  en.value(instancename.vElyInstalledMW) # Electrolyzer capacity (MWe)
    df1['H2St_tonne'] =  en.value(instancename.vH2StInstalledNumber*instancename.pH2kgpertank/1000) #  H2 storage size (tonne)
    df1['Comp_MW'] =  en.value(instancename.vCompInstalledMW) # Compressor capacity (MWe)
    df1['Inv_MW'] = en.value(instancename.vInvInstalledMW)
    
    df1['H2ProdCost_perkg'] =  en.value(instancename.eSysTotalCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg cost of H2
    #  Breakdown of system cost
#    m.ePVFixCost + m.eStFixCost + m.eElyFixCost + m.eH2StFixCost +m.eSysVOMCost
    df1['PVFixCost_perkg'] =  en.value(instancename.ePVFixCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg cost of H2
                                
    df1['StFixCost_perkg'] =  en.value(instancename.eStFixCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg cost of H2
    
    df1['ElyFixCost_perkg'] =  en.value(instancename.eElyFixCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg    
                                        
    df1['H2StFixCost_perkg'] =  en.value(instancename.eH2StFixCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg    
    
    df1['SysVOMCost_perkg'] =  en.value(instancename.eSysVOMCost)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg    
                                        
    df1['GridRevenue_perkg'] =  -en.value(instancename.eGridRevenue)/ \
                                        sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps]) #  $/kg    
    
    
    df1['H2ProdCost_perGJ'] =  en.value(instancename.eSysTotalCost)/ \
                                       sum([en.value(instancename.vTotalH2Output[t]) for t in Timesteps])/en.value(instancename.pH2LHV)*1000 #  $/GJ LHV cost of H2

    df1['SlackCost'] = en.value(instancename.pProductionSlackCost)
    
    df1['ProductionSlack'] = en.value(instancename.eProductionSlack)
    
    #  PV curtailment as a fraction
    df1['PV_curtailment'] =1 -sum([en.value(instancename.vPVOutput[t]) for t in Timesteps])/ \
                                                                sum([en.value(instancename.pPVCapFactor[t]*instancename.vPVInstalledMW) for t in Timesteps])
                                        
    df1['ElyCF']= \
     sum([en.value(instancename.vH2flowProd[t]) +en.value(instancename.vH2flowStorage[t]) for t in Timesteps])/ \
     (en.value(instancename.vElyInstalledMW)*1000/en.value(instancename.pElySpecPower)*len(instancename.t.data())) # Capacity factor for electrolyzer
     
    df1['ElyToPVCap']= \
        en.value(instancename.vElyInstalledMW)/en.value(instancename.vPVInstalledMW) # Ratio of Electrolyzer to PV capacity
    
    return df1
