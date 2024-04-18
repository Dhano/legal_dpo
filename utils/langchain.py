from typing import List
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import tiktoken

tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')

class Entity(BaseModel):
    value: str = Field(description="Entity value")
    type: str = Field(description="Entity type")

class Triplet(BaseModel):
    head: Entity = Field(description="Head entity and its type")
    relationship: str = Field(description="Relationship")
    tail: Entity = Field(description="Tail entity and its type")

class Response(BaseModel):
    triplets: List[Triplet] = Field(description="List of triplets") 

parser = PydanticOutputParser(pydantic_object=Response)

# I will provide you with a legal document. Your task is to Understand the document and create a detailed summary about the facts in the document, extract facts from summary in the form of triplets
#                     for constructing a knowledge graph. The knowledge graph should be comprehensive and dense to facilitate legal analysis,
#                     focus more on extracting  essential information and omitting obvious triplets.

langchain_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
                    As an Indian lawyer, your job is to understand legal documents. Right now, you're building a detailed knowledge graph based on information in a given legal document. 
                    It's crucial that this graph includes all the fact, evidences, observations  from the document,
                    so nothing important is left out.
                    The goal is to make legal analysis easier by focusing on the key information and skipping the obvious stuff.
                  
                    Each triplet should be in the form of (h:type, r, o:type), where 'h' stands for the head entity,
                    'r' for the relationship, and 'o' for the tail entity. The 'type' denotes the category of the corresponding entity.

                    The Entities should be non-generic and can be classified into the following categories:
                    - Actor / Player: A person who has a role in a legal matter (e.g., Buyer, Provider, Lawyer, Law Firm, Expert, Employer, Employee, Buyer, Seller, Lessor, Lessee, Debtor, Creditor, Payor, Payee, Landlord, Tenant).
                    - Area of Law: The practice area into which a legal matter or legal area of study falls (e.g., Criminal Law, Real Property Law, Mergers and Acquisitions Law, Personal and Family Law, Tax and Revenue Law).
                    - Asset Type: Type of resource that is owned or controlled by a person, business, or economic entity
                    - Communication Modality: Entities' chosen communication method (e.g., written, email, telephone, portal), as well as time (e.g., synchronous, asynchronous).
                    - Currency: A standardization of money that is used, circulated, or exchanged (e.g., banknotes, coins).
                    - Document / Artifact: A written, drawn, presented, or memorialized representation of thought or expression, including evidence such as recordings and other artifacts.
                    - Engagement Terms: Terms to define an engagement for providing legal services.
                    - Event: A matter's events, as well as collections of those events (often noted as "phases").
                    - Forums and Venues: Organization or government entity that administers proceedings.
                    - Governmental Body: Administrative entities of government or state agency or appointed commission, as a permanent or semi-permanent governmental organization that oversees or administers specific governmental functions.
                    - Industry: An economic branch that produces a related set of raw materials, goods, or services (e.g., Agriculture Industry, Pharmaceuticals Industry).
                    - Legal Authorities: Documents or publications that guide legal rights and obligations (e.g., caselaw, statutes, regulations, rules) or that can be cited as providing guidance on the law (e.g., secondary legal authorities).
                    - Legal Entity: A person, company, organization, or other entity that has legal rights and obligations.
                    - Location: The name of a position on the Earth, usually in the context of continents, countries, and their political subdivisions (e.g., regions, states or provinces, cities, towns, villages).
                    - Matter Narrative: A textual narrative of a matter's factual and legal details.
                    - Objectives: Specific aims, goals, arguments, plans, intentions, designs, purposes, schemes, etc. that are constructed by a party in a legal matter, and the legal or other professional frameworks that support their execution.
                    - Service: The legal work performed, usually by a Legal Services Provider, in the course of a legal matter.
                    - Status: The state or condition of a proceeding, legal element, or legal matter (e.g., open, closed, canceled, expired).

                    The Relationships r between these entities must be represented by meaningful verbs/actions and its  properties  like cause purpose manner etc .

                    Remember to conduct entity disambiguation, consolidating different phrases or acronyms that refer to the same entity 
                    Simplify each entity of the triplet to be no more than three four word.

                    Include triplets that are implicitly inferred from the document's context but not explicitly 
                    stated, in order to ensure the graph is both connected and dense

                    """+
                    parser.get_format_instructions()
                    +"""
                    
                    Now, let's apply this process to the following Document:   
                """
                
            )
        ),
        HumanMessagePromptTemplate.from_template("<Document>{document}</Document>"),
    ]
)


def trim_document(query_text: str) -> str:
    #default gpt triming
    query_text = query_text[:8500].replace("\n", " ")
      
    while len(tokenizer.encode(str(langchain_chat_template.format_messages(document=query_text)))) > 3548:
        query_text = query_text[:len(query_text)-500]

    return query_text
    


examples = """
For refrence consider the following part of a document and the corresponding triplets extracted from it:

                    Example 1: 

                    5. On careful perusal of the material evidence placed on record, it would go to show that complainant
                    is a Class-1 Civil Contractor and permanent resident of GPE. 
                    Accused No.1 is the Private Limited Company, accused Nos.2 and 3 are husband and wife and
                    are the Directors of accused No.1 Company. Accused Nos.2 and 3 are doing software business 
                    through accused No.1 Company. 
                    On DATE, accused Nos.2 and 3 visited the complainant at his place and sought financial assistance 
                    of Rs. 5 lakhs for the purpose of improving the business of accused No.1 Company. 

                    Head Entity: 'Complainant' Type: Actor / Player
                    Relationship: 'is'
                    Tail Entity: 'Class-1 Civil Contractor' Type: Legal Entity

                    Head Entity: 'Complainant' Type: Actor / Player
                    Relationship: 'is permanent resident of'
                    Tail Entity: 'GPE' Type: Location

                    Head Entity: 'Accused Nos.2 and 3' Type: Actor / Player
                    Relationship: 'are'
                    Tail Entity: 'husband and wife' Type: Actor / Player

                    Head Entity: 'Accused Nos.2 and 3' Type: Actor / Player
                    Relationship: 'Directors of'
                    Tail Entity: 'Private Limited Company' Type: Industry

                    Head Entity: 'Accused Nos.2 and 3' Type: Actor / Player
                    Relationship: 'are doing'
                    Tail Entity: 'Software business' Type: Industry

                    Head Entity: 'Accused Nos.2 and 3' Type: Actor / Player
                    Relationship: 'visited'
                    Tail Entity: 'Complainant' Type: Actor / Player

                    Head Entity: 'Accused Nos.2 and 3' Type: Actor / Player
                    Relationship: 'sought financial assistance For improving'
                    Tail Entity: 'business' Type: Industry

                    Example 2:

                    11. So far as facts of the present case are concerned, the NEUTRAL CITATION R/CR.MA/3585/2024 ORDER DATED:
                    DATE undefined prosecution had sought to lead the evidence by examining the witnesses to prove that the 
                    deceased had committed suicide because of the mental and physical harassment of the appellants- accused.
                    The PW-1 WITNESS, who happened to be the mother though had alleged in her examination-in-chief that 
                    her daughter was murdered by the accused by throwing her in the well, she had admitted that when she 
                    reached at the spot, she had not seen the dead body of her daughter in the well. 
                    She had also admitted that she had not stated in her complaint that her daughter had committed suicide
                    by jumping into the well on account of the mental and physical harassment caused by the accused.

                    Head Entity: 'Prosecution' Type: Actor / Player
                    Relationship: 'sought to lead'
                    Tail Entity: 'Evidence' Type: Document / Artifact

                    Head Entity: 'Prosecution' Type: Actor / Player
                    Relationship: 'to prove committed suicide'
                    Tail Entity: 'Deceased' Type: Actor / Player

                    Head Entity: 'Deceased ' Type: Event
                    Relationship: 'committed suicide because of Mental and physical harassment by'
                    Tail Entity: 'accused' Type: Actor / Player

                    Head Entity: 'PW-1 Witness' Type: Actor / Player
                    Relationship: 'is Mother of'
                    Tail Entity: 'deceased' Type: Actor / Player

                    Head Entity: 'PW-1 Witness' Type: Actor / Player
                    Relationship: 'alleged'
                    Tail Entity: 'Accused murdered her daughter by throwing her in well' Type: Event

                    Head Entity: 'PW-1 Witness' Type: Actor / Player
                    Relationship: 'admitted not seeing Dead body in the well of'
                    Tail Entity: 'daughter ' Type: Event

                    Head Entity: 'PW-1 Witness' Type: Actor / Player
                    Relationship: 'admitted'
                    Tail Entity: 'not stating in complaint Daughter committed suicide due to harassment' Type: Event
                    
"""


