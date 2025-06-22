import pandas as pd  
  
# Create index data  
data = [  
    {  
        "_id_": {  
            "v": 2,  
            "key": [("_id", 1)],  
        },  
    },  
    {  
        "company_1_wireAccount_1": {  
            "v": 2,  
            "key": [("company", 1), ("wireAccount", 1)],  
        },  
    },  
    {  
        "_cls_1_purpose_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("purpose", 1)],  
        },  
    },  
    {  
        "company_1__cls_1_run_1_role_1_eeBankAccount_1": {  
            "v": 2,  
            "key": [("company", 1), ("_cls", 1), ("run", 1), ("role", 1), ("eeBankAccount", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"eeBankAccount": {"$exists": True}},  
                    {"_cls": "LedgerEntry.LedgerEntryWithMandatoryCompany.LedgerEntryToEmployee"},  
                    {"run": {"$exists": True}},  
                    {"retries": {"$eq": 0}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "company_1_run_1": {  
            "v": 1,  
            "key": [("company", 1), ("run", 1)],  
        },  
    },  
    {  
        "_cls_1_checkDate_1_method_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("checkDate", 1), ("method", 1)],  
        },  
    },  
    {  
        "_cls_1_createdAt_1_state_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("createdAt", 1), ("state", 1)],  
            "partialFilterExpression": {"isDeleted": False},  
        },  
    },  
    {  
        "company_1_priorPayrollTaxesDraft_1": {  
            "v": 2,  
            "key": [("company", 1), ("priorPayrollTaxesDraft", 1)],  
        },  
    },  
    {  
        "company_1_companyBankAccount_1": {  
            "v": 2,  
            "key": [("company", 1), ("companyBankAccount", 1)],  
        },  
    },  
    {  
        "_cls_1__id_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("_id", 1)],  
            "partialFilterExpression": {"isDeleted": {"$eq": False}},  
        },  
    },  
    {  
        "_cls_1_ach_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("ach", 1)],  
            "partialFilterExpression": {"ach": {"$exists": True}},  
        },  
    },  
    {  
        "_cls_1_checkDate_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("checkDate", 1)],  
            "partialFilterExpression": {  
                "$and": [  
                    {"state": {"$eq": "PENDING"}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "company_1_employee_1": {  
            "v": 2,  
            "key": [("company", 1), ("employee", 1)],  
        },  
    },  
    {  
        "company_1_ach_1": {  
            "v": 1,  
            "key": [("company", 1), ("ach", 1)],  
        },  
    },  
    {  
        "_cls_1_run_1_purpose_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("run", 1), ("purpose", 1)],  
        },  
    },  
    {  
        "company_1__cls_1_purpose_1": {  
            "v": 2,  
            "key": [("company", 1), ("_cls", 1), ("purpose", 1)],  
        },  
    },  
    {  
        "company_1__cls_1_run_1_purpose_1": {  
            "v": 2,  
            "key": [("company", 1), ("_cls", 1), ("run", 1), ("purpose", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"run": {"$exists": True}},  
                    {"isIAT": {"$eq": False}},  
                    {"_cls": "LedgerEntry.LedgerEntryWithMandatoryCompany.LedgerEntryFromCompany"},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "_cls_1_eeBankAccount_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("eeBankAccount", 1)],  
            "partialFilterExpression": {"eeBankAccount": {"$exists": True}},  
        },  
    },  
    {  
        "_cls_1_wireAccount_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("wireAccount", 1)],  
            "partialFilterExpression": {"wireAccount": {"$exists": True}},  
        },  
    },  
    {  
        "_cls_1_pendingIncomingWireId_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("pendingIncomingWireId", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"pendingIncomingWireId": {"$exists": True}},  
                    {"_cls": "LedgerEntry.LedgerEntryWithMandatoryCompany.LedgerEntryToCompany"},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "company_1_role_1": {  
            "v": 2,  
            "key": [("company", 1), ("role", 1)],  
        },  
    },  
    {  
        "company_1_achCreditAgencyPaymentsRef_1": {  
            "v": 2,  
            "key": [("company", 1), ("achCreditAgencyPaymentsRef", 1)],  
        },  
    },  
    {  
        "checkDate_cls_triggeredComplianceScreeningTypes": {  
            "v": 2,  
            "key": [("_cls", 1), ("checkDate", 1), ("triggeredComplianceScreeningTypes", 1)],  
            "background": True,  
        },  
    },  
    {  
        "company_1_eeBankAccount_1": {  
            "v": 2,  
            "key": [("company", 1), ("eeBankAccount", 1)],  
        },  
    },  
    {  
        "company_1__cls_1_priorPayrollTaxesDraft_1": {  
            "v": 2,  
            "key": [("company", 1), ("_cls", 1), ("priorPayrollTaxesDraft", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"priorPayrollTaxesDraft": {"$exists": True}},  
                    {"_cls": "LedgerEntry.LedgerEntryWithMandatoryCompany.LedgerEntryFromCompany"},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "_cls_1_ledgerUniqueKey_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("ledgerUniqueKey", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"ledgerUniqueKey": {"$exists": True}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "_cls_1_achCreditAgencyPaymentsRef_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("achCreditAgencyPaymentsRef", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"achCreditAgencyPaymentsRef": {"$exists": True}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "company_1_invoice_1": {  
            "v": 2,  
            "key": [("company", 1), ("invoice", 1)],  
        },  
    },  
    {  
        "company_1__cls_1": {  
            "v": 2,  
            "key": [("company", 1), ("_cls", 1)],  
            "partialFilterExpression": {  
                "$and": [  
                    {"company": {"$exists": True}},  
                    {"_cls": {"$exists": True}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
    {  
        "_cls_1_invoice_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("invoice", 1)],  
            "partialFilterExpression": {"invoice": {"$exists": True}},  
        },  
    },  
    {  
        "company_1_wire_1": {  
            "v": 2,  
            "key": [("company", 1), ("wire", 1)],  
        },  
    },  
    {  
        "_cls_1_wire_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("wire", 1)],  
        },  
    },  
    {  
        "_cls_1_idNumber_1": {  
            "v": 2,  
            "key": [("_cls", 1), ("idNumber", 1)],  
            "unique": True,  
            "partialFilterExpression": {  
                "$and": [  
                    {"idNumber": {"$exists": True}},  
                    {"isDeleted": {"$eq": False}},  
                ]  
            },  
        },  
    },  
]  

  
# Preprocess the data to extract relevant columns
formatted_data = []
for index in data:
    for index_name, index_details in index.items():
        formatted_data.append({
            "Index Name": index_name,
            "Key Fields": ", ".join([f"{field[0]}: {field[1]}" for field in index_details.get("key", [])]),
            "Unique": index_details.get("unique", False),
            "Partial Filter Expression": str(index_details.get("partialFilterExpression", None)),
            "Background": index_details.get("background", False),
            "Version": index_details.get("v", None),
        })

# Convert to DataFrame
df = pd.DataFrame(formatted_data)

# Save to Excel with formatting
excel_file = "ledger_entry_indexes.xlsx"
with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Indexes")

    # Access the workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets["Indexes"]

    # Set column widths for better readability
    worksheet.set_column("A:A", 30)  # Index Name
    worksheet.set_column("B:B", 50)  # Key Fields
    worksheet.set_column("C:C", 10)  # Unique
    worksheet.set_column("D:D", 70)  # Partial Filter Expression
    worksheet.set_column("E:E", 15)  # Background
    worksheet.set_column("F:F", 10)  # Version

print(f"Excel file '{excel_file}' created successfully!")