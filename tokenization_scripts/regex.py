import re

regex = r"""
# Adapted from DCML at https://github.com/DCMLab/standards/blob/docs/harmony.py
^                                          
(\{)?\.?                                   
((?P<globalkey>[a-gA-G]([-+]*))\.)?       
((?P<localkey>([-+]*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?  
((?P<pedal>([-+]*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?   
(?P<chord>                                 
    (?P<numeral>([+=-]*)(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none)[-+=]*)? 
    (?P<form>(%|o|\+|M|\+M))?             
    (?P<figbass>(7|65|43|42|2|64|6))?     
    (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d+)+)\))?  
    (/(?P<relativeroot>((b*|\#*)([-+]*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?  
)
(?P<pedalend>\])?                          
(\|(?P<cadence>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?  
(?P<phraseend>(\\\\|\{|\}|\}\{))?         
$                                           
"""

regex = re.compile(regex, re.VERBOSE)