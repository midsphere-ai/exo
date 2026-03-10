"""Multi-criteria answer evaluation system.

Ported from node-DeepResearch's src/tools/evaluator.ts.
Provides 6-criteria evaluation: definitive, freshness, plurality,
completeness, attribution, and strict.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger("deepsearch")

# Type alias for the generate function
# async def(schema: type[BaseModel], system: str, prompt: str) -> BaseModel
GenerateFn = Callable[..., Awaitable[BaseModel]]

# Type alias for schema generator callable
# schema_gen(eval_type: str) -> type[BaseModel]
SchemaGenFn = Callable[..., type[BaseModel]]

TOOL_NAME = "evaluator"


# ---------------------------------------------------------------------------
# Prompt helpers — each returns (system_prompt, user_prompt)
# ---------------------------------------------------------------------------


def _get_reject_all_answers_prompt(
    question: str,
    answer: Any,  # AnswerAction
    all_knowledge: list[Any],  # list[KnowledgeItem]
) -> tuple[str, str]:
    """Build the strict-mode evaluation prompt."""
    from ..utils.text_tools import get_knowledge_str

    knowledge_str = get_knowledge_str(all_knowledge)

    system = (
        """You are a ruthless and picky answer evaluator trained to REJECT answers. You can't stand any shallow answers.
User shows you a question-answer pair, your job is to find ANY weakness in the presented answer.
Identity EVERY missing detail.
First, argue AGAINST the answer with the strongest possible case.
Then, argue FOR the answer.
Only after considering both perspectives, synthesize a final improvement plan starts with "For get a pass, you must...".
Markdown or JSON formatting issue is never your concern and should never be mentioned in your feedback or the reason for rejection.

You always endorse answers in most readable natural language format.
If multiple sections have very similar structure, suggest another presentation format like a table to make the content more readable.
Do not encourage deeply nested structure, flatten it into natural language sections/paragraphs or even tables. Every table should use HTML table syntax <table> <thead> <tr> <th> <td> without any CSS styling.

The following knowledge items are provided for your reference. Note that some of them may not be directly related to the question/answer user provided, but may give some subtle hints and insights:
"""
        + "\n\n".join(knowledge_str)
    )

    user = f"""Dear reviewer, I need your feedback on the following question-answer pair:

<question>
{question}
</question>

Here is my answer for the question:
<answer>
{answer.answer}
</answer>

Could you please evaluate it based on your knowledge and strict standards? Let me know how to improve it."""

    return system, user


def _get_definitive_prompt(question: str, answer: str) -> tuple[str, str]:
    """Build the definitive evaluation prompt."""
    system = """You are an evaluator of answer definitiveness. Analyze if the given answer provides a definitive response or not.

<rules>
First, if the answer is not a direct response to the question, it must return false.

Definitiveness means providing a clear, confident response. The following approaches are considered definitive:
  1. Direct, clear statements that address the question
  2. Comprehensive answers that cover multiple perspectives or both sides of an issue
  3. Answers that acknowledge complexity while still providing substantive information
  4. Balanced explanations that present pros and cons or different viewpoints

The following types of responses are NOT definitive and must return false:
  1. Expressions of personal uncertainty: "I don't know", "not sure", "might be", "probably"
  2. Lack of information statements: "doesn't exist", "lack of information", "could not find"
  3. Inability statements: "I cannot provide", "I am unable to", "we cannot"
  4. Negative statements that redirect: "However, you can...", "Instead, try..."
  5. Non-answers that suggest alternatives without addressing the original question

Note: A definitive answer can acknowledge legitimate complexity or present multiple viewpoints as long as it does so with confidence and provides substantive information directly addressing the question.
</rules>

<examples>
Question: "What are the system requirements for running Python 3.9?"
Answer: "I'm not entirely sure, but I think you need a computer with some RAM."
Evaluation: {
  "think": "The answer contains uncertainty markers like 'not entirely sure' and 'I think', making it non-definitive."
  "pass": false,
}

Question: "What are the system requirements for running Python 3.9?"
Answer: "Python 3.9 requires Windows 7 or later, macOS 10.11 or later, or Linux."
Evaluation: {
  "think": "The answer makes clear, definitive statements without uncertainty markers or ambiguity."
  "pass": true,
}

Question: "Who will be the president of the United States in 2032?"
Answer: "I cannot predict the future, it depends on the election results."
Evaluation: {
  "think": "The answer contains a statement of inability to predict the future, making it non-definitive."
  "pass": false,
}

Question: "Who is the sales director at Company X?"
Answer: "I cannot provide the name of the sales director, but you can contact their sales team at sales@companyx.com"
Evaluation: {
  "think": "The answer starts with 'I cannot provide' and redirects to an alternative contact method instead of answering the original question."
  "pass": false,
}

Question: "what is the twitter account of jina ai's founder?"
Answer: "The provided text does not contain the Twitter account of Jina AI's founder."
Evaluation: {
  "think": "The answer indicates a lack of information rather than providing a definitive response."
  "pass": false,
}

Question: "\u91cf\u5b50\u30b3\u30f3\u30d4\u30e5\u30fc\u30bf\u306e\u8a08\u7b97\u80fd\u529b\u3092\u5177\u4f53\u7684\u306b\u6e2c\u5b9a\u3059\u308b\u65b9\u6cd5\u306f\u4f55\u3067\u3059\u304b\uff1f"
Answer: "\u91cf\u5b50\u30b3\u30f3\u30d4\u30e5\u30fc\u30bf\u306e\u8a08\u7b97\u80fd\u529b\u306f\u91cf\u5b50\u30d3\u30c3\u30c8\uff08\u30ad\u30e5\u30fc\u30d3\u30c3\u30c8\uff09\u306e\u6570\u3001\u30b2\u30fc\u30c8\u5fe0\u5b9f\u5ea6\u3001\u30b3\u30d2\u30fc\u30ec\u30f3\u30b9\u6642\u9593\u3067\u6e2c\u5b9a\u3055\u308c\u307e\u3059\u3002"
Evaluation: {
  "think": "The answer provides specific, definitive metrics for measuring quantum computing power without uncertainty markers or qualifications."
  "pass": true,
}

Question: "\u5982\u4f55\u8bc1\u660e\u54e5\u5fb7\u5df4\u8d6b\u731c\u60f3\u662f\u6b63\u786e\u7684\uff1f"
Answer: "\u76ee\u524d\u5c1a\u65e0\u5b8c\u6574\u8bc1\u660e\uff0c\u4f462013\u5e74\u5f20\u76ca\u5510\u8bc1\u660e\u4e86\u5b58\u5728\u65e0\u7a77\u591a\u5bf9\u76f8\u5dee\u4e0d\u8d85\u8fc77000\u4e07\u7684\u7d20\u6570\uff0c\u540e\u6765\u8fd9\u4e2a\u754c\u88ab\u7f29\u5c0f\u5230246\u3002"
Evaluation: {
  "think": "The answer begins by stating no complete proof exists, which is a non-definitive response, and then shifts to discussing a related but different theorem about bounded gaps between primes."
  "pass": false,
}

Question: "Wie kann man mathematisch beweisen, dass P \u2260 NP ist?"
Answer: "Ein Beweis f\u00fcr P \u2260 NP erfordert, dass man zeigt, dass mindestens ein NP-vollst\u00e4ndiges Problem nicht in polynomieller Zeit l\u00f6sbar ist. Dies k\u00f6nnte durch Diagonalisierung, Schaltkreiskomplexit\u00e4t oder relativierende Barrieren erreicht werden."
Evaluation: {
  "think": "The answer provides concrete mathematical approaches to proving P \u2260 NP without uncertainty markers, presenting definitive methods that could be used."
  "pass": true,
}

Question: "Is universal healthcare a good policy?"
Answer: "Universal healthcare has both advantages and disadvantages. Proponents argue it provides coverage for all citizens, reduces administrative costs, and leads to better public health outcomes. Critics contend it may increase wait times, raise taxes, and potentially reduce innovation in medical treatments. Most developed nations have implemented some form of universal healthcare with varying structures and degrees of coverage."
Evaluation: {
  "think": "The answer confidently presents both sides of the debate with specific points for each perspective. It provides substantive information directly addressing the question without expressions of personal uncertainty."
  "pass": true,
}

Question: "Should companies use AI for hiring decisions?"
Answer: "There are compelling arguments on both sides of this issue. Companies using AI in hiring can benefit from reduced bias in initial screening, faster processing of large applicant pools, and potentially better matches based on skills assessment. However, these systems can also perpetuate historical biases in training data, may miss nuanced human qualities, and raise privacy concerns. The effectiveness depends on careful implementation, human oversight, and regular auditing of these systems."
Evaluation: {
  "think": "The answer provides a balanced, detailed examination of both perspectives on AI in hiring. It acknowledges complexity while delivering substantive information with confidence."
  "pass": true,
}

Question: "Is nuclear energy safe?"
Answer: "I'm not an expert on energy policy, so I can't really say if nuclear energy is safe or not. There have been some accidents but also many successful plants."
Evaluation: {
  "think": "The answer contains explicit expressions of personal uncertainty ('I'm not an expert', 'I can't really say') and provides only vague information without substantive content."
  "pass": false,
}
</examples>"""

    user = f"""
Question: {question}
Answer: {answer}"""

    return system, user


def _get_freshness_prompt(
    question: str,
    answer: Any,  # AnswerAction
    current_time: str,
) -> tuple[str, str]:
    """Build the freshness evaluation prompt."""
    system = f"""You are an evaluator that analyzes if answer content is likely outdated based on mentioned dates (or implied datetime) and current system time: {current_time}

<rules>
Question-Answer Freshness Checker Guidelines

| QA Type                  | Max Age (Days) | Notes                                                                 |
|--------------------------|--------------|-----------------------------------------------------------------------|
| Financial Data (Real-time)| 0.1        | Stock prices, exchange rates, crypto (real-time preferred)             |
| Breaking News            | 1           | Immediate coverage of major events                                     |
| News/Current Events      | 1           | Time-sensitive news, politics, or global events                        |
| Weather Forecasts        | 1           | Accuracy drops significantly after 24 hours                            |
| Sports Scores/Events     | 1           | Live updates required for ongoing matches                              |
| Security Advisories      | 1           | Critical security updates and patches                                  |
| Social Media Trends      | 1           | Viral content, hashtags, memes                                         |
| Cybersecurity Threats    | 7           | Rapidly evolving vulnerabilities/patches                               |
| Tech News                | 7           | Technology industry updates and announcements                          |
| Political Developments   | 7           | Legislative changes, political statements                              |
| Political Elections      | 7           | Poll results, candidate updates                                        |
| Sales/Promotions         | 7           | Limited-time offers and marketing campaigns                            |
| Travel Restrictions      | 7           | Visa rules, pandemic-related policies                                  |
| Entertainment News       | 14          | Celebrity updates, industry announcements                              |
| Product Launches         | 14          | New product announcements and releases                                 |
| Market Analysis          | 14          | Market trends and competitive landscape                                |
| Competitive Intelligence | 21          | Analysis of competitor activities and market position                  |
| Product Recalls          | 30          | Safety alerts or recalls from manufacturers                            |
| Industry Reports         | 30          | Sector-specific analysis and forecasting                               |
| Software Version Info    | 30          | Updates, patches, and compatibility information                        |
| Legal/Regulatory Updates | 30          | Laws, compliance rules (jurisdiction-dependent)                        |
| Economic Forecasts       | 30          | Macroeconomic predictions and analysis                                 |
| Consumer Trends          | 45          | Shifting consumer preferences and behaviors                            |
| Scientific Discoveries   | 60          | New research findings and breakthroughs (includes all scientific research) |
| Healthcare Guidelines    | 60          | Medical recommendations and best practices (includes medical guidelines)|
| Environmental Reports    | 60          | Climate and environmental status updates                               |
| Best Practices           | 90          | Industry standards and recommended procedures                          |
| API Documentation        | 90          | Technical specifications and implementation guides                     |
| Tutorial Content         | 180         | How-to guides and instructional materials (includes educational content)|
| Tech Product Info        | 180         | Product specs, release dates, or pricing                               |
| Statistical Data         | 180         | Demographic and statistical information                                |
| Reference Material       | 180         | General reference information and resources                            |
| Historical Content       | 365         | Events and information from the past year                              |
| Cultural Trends          | 730         | Shifts in language, fashion, or social norms                           |
| Entertainment Releases   | 730         | Movie/TV show schedules, media catalogs                                |
| Factual Knowledge        | \u221e           | Static facts (e.g., historical events, geography, physical constants)   |

### Implementation Notes:
1. **Contextual Adjustment**: Freshness requirements may change during crises or rapid developments in specific domains.
2. **Tiered Approach**: Consider implementing urgency levels (critical, important, standard) alongside age thresholds.
3. **User Preferences**: Allow customization of thresholds for specific query types or user needs.
4. **Source Reliability**: Pair freshness metrics with source credibility scores for better quality assessment.
5. **Domain Specificity**: Some specialized fields (medical research during pandemics, financial data during market volatility) may require dynamically adjusted thresholds.
6. **Geographic Relevance**: Regional considerations may alter freshness requirements for local regulations or events.
</rules>"""

    user = f"""
Question: {question}
Answer:
{json.dumps(answer.model_dump() if hasattr(answer, "model_dump") else answer, ensure_ascii=False)}

Please look at my answer and references and think.
"""

    return system, user


def _get_completeness_prompt(question: str, answer: str) -> tuple[str, str]:
    """Build the completeness evaluation prompt."""
    system = """You are an evaluator that determines if an answer addresses all explicitly mentioned aspects of a multi-aspect question.

<rules>
For questions with **explicitly** multiple aspects:

1. Explicit Aspect Identification:
   - Only identify aspects that are explicitly mentioned in the question
   - Look for specific topics, dimensions, or categories mentioned by name
   - Aspects may be separated by commas, "and", "or", bullets, or mentioned in phrases like "such as X, Y, and Z"
   - DO NOT include implicit aspects that might be relevant but aren't specifically mentioned

2. Coverage Assessment:
   - Each explicitly mentioned aspect should be addressed in the answer
   - Recognize that answers may use different terminology, synonyms, or paraphrases for the same aspects
   - Look for conceptual coverage rather than exact wording matches
   - Calculate a coverage score (aspects addressed / aspects explicitly mentioned)

3. Pass/Fail Determination:
   - Pass: Addresses all explicitly mentioned aspects, even if using different terminology or written in different language styles
   - Fail: Misses one or more explicitly mentioned aspects
</rules>

<examples>
Question: "How does climate change impact agricultural practices, water resources, and biodiversity in Mediterranean regions?"
Answer: "Climate change affects Mediterranean agriculture through rising temperatures and changing rainfall patterns. Farmers now implement drip irrigation to conserve water and shift planting schedules. Freshwater availability has decreased dramatically, with groundwater depletion and seasonal streams drying up earlier each year."
Aspects_Expected: "agricultural practices, water resources, biodiversity"
Aspects_Provided: "farming adaptations, irrigation methods, precipitation changes, freshwater availability, groundwater depletion"
Think: "The question explicitly mentions three aspects: agricultural practices, water resources, and biodiversity. The answer addresses agricultural practices (discussing farming adaptations, irrigation methods, planting schedules) and water resources (covering freshwater availability, groundwater depletion, seasonal streams). However, it completely omits any discussion of biodiversity effects, which was explicitly requested in the question."
Pass: false

Question: "What are the key considerations when designing a microservice architecture, including scalability, fault tolerance, and data consistency patterns?"
Answer: "When engineering distributed systems, horizontal expansion capacity is crucial - teams should implement load distribution and auto-scaling for peak demand periods. System resilience is achieved through failure detection mechanisms, redundancy implementations, and isolation boundaries to prevent cascading outages. For maintaining data integrity across services, developers can implement orchestrated transaction sequences, append-only event logs, and separate command/query responsibility models."
Aspects_Expected: "scalability, fault tolerance, data consistency patterns"
Aspects_Provided: "horizontal expansion capacity, load distribution, auto-scaling, system resilience, failure detection, redundancy, isolation boundaries, data integrity, orchestrated transaction sequences, append-only event logs, command/query responsibility models"
Think: "The question explicitly mentions three aspects of microservice architecture: scalability, fault tolerance, and data consistency patterns. Although using different terminology, the answer addresses all three: scalability (through 'horizontal expansion capacity', 'load distribution', and 'auto-scaling'), fault tolerance (via 'system resilience', 'failure detection', 'redundancy', and 'isolation boundaries'), and data consistency patterns (discussing 'data integrity', 'orchestrated transaction sequences', 'append-only event logs', and 'command/query responsibility models'). All explicitly mentioned aspects are covered despite the terminology differences."
Pass: true

Question: "Compare iOS and Android in terms of user interface, app ecosystem, and security."
Answer: "Apple's mobile platform presents users with a curated visual experience emphasizing minimalist design and consistency, while Google's offering focuses on flexibility and customization options. The App Store's review process creates a walled garden with higher quality control but fewer options, whereas Play Store offers greater developer freedom and variety. Apple employs strict sandboxing techniques and maintains tight hardware-software integration."
Aspects_Expected: "user interface, app ecosystem, security"
Aspects_Provided: "visual experience, minimalist design, flexibility, customization, App Store review process, walled garden, quality control, Play Store, developer freedom, sandboxing, hardware-software integration"
Think: "The question explicitly asks for a comparison of iOS and Android across three specific aspects: user interface, app ecosystem, and security. The answer addresses user interface (discussing 'visual experience', 'minimalist design', 'flexibility', and 'customization') and app ecosystem (mentioning 'App Store review process', 'walled garden', 'quality control', 'Play Store', and 'developer freedom'). For security, it mentions 'sandboxing' and 'hardware-software integration', which are security features of iOS, but doesn't provide a comparative analysis of Android's security approach. Since security is only partially addressed for one platform, the comparison of this aspect is incomplete."
Pass: false

Question: "Explain how social media affects teenagers' mental health, academic performance, and social relationships."
Answer: "Platforms like Instagram and TikTok have been linked to psychological distress among adolescents, with documented increases in comparative thinking patterns and anxiety about social exclusion. Scholastic achievement often suffers as screen time increases, with homework completion rates declining and attention spans fragmenting during study sessions. Peer connections show a complex duality - digital platforms facilitate constant contact with friend networks while sometimes diminishing in-person social skill development and enabling new forms of peer harassment."
Aspects_Expected: "mental health, academic performance, social relationships"
Aspects_Provided: "psychological distress, comparative thinking, anxiety about social exclusion, scholastic achievement, screen time, homework completion, attention spans, peer connections, constant contact with friend networks, in-person social skill development, peer harassment"
Think: "The question explicitly asks about three aspects of social media's effects on teenagers: mental health, academic performance, and social relationships. The answer addresses all three using different terminology: mental health (discussing 'psychological distress', 'comparative thinking', 'anxiety about social exclusion'), academic performance (mentioning 'scholastic achievement', 'screen time', 'homework completion', 'attention spans'), and social relationships (covering 'peer connections', 'constant contact with friend networks', 'in-person social skill development', and 'peer harassment'). All explicitly mentioned aspects are covered despite using different language."
Pass: true

Question: "What economic and political factors contributed to the 2008 financial crisis?"
Answer: "The real estate market collapse after years of high-risk lending practices devastated mortgage-backed securities' value. Wall Street had created intricate derivative products that disguised underlying risk levels, while credit assessment organizations failed in their oversight role. Legislative changes in the financial industry during the 1990s eliminated regulatory guardrails that previously limited excessive leverage and speculation among investment banks."
Aspects_Expected: "economic factors, political factors"
Aspects_Provided: "real estate market collapse, high-risk lending, mortgage-backed securities, derivative products, risk disguising, credit assessment failures, legislative changes, regulatory guardrail elimination, leverage, speculation"
Think: "The question explicitly asks about two categories of factors: economic and political. The answer addresses economic factors ('real estate market collapse', 'high-risk lending', 'mortgage-backed securities', 'derivative products', 'risk disguising', 'credit assessment failures') and political factors ('legislative changes', 'regulatory guardrail elimination'). While using different terminology, the answer covers both explicitly requested aspects."
Pass: true

Question: "\u30b3\u30ed\u30ca\u30a6\u30a4\u30eb\u30b9\u306e\u611f\u67d3\u62e1\u5927\u304c\u7d4c\u6e08\u3001\u6559\u80b2\u30b7\u30b9\u30c6\u30e0\u3001\u304a\u3088\u3073\u533b\u7642\u30a4\u30f3\u30d5\u30e9\u306b\u3069\u306e\u3088\u3046\u306a\u5f71\u97ff\u3092\u4e0e\u3048\u307e\u3057\u305f\u304b\uff1f"
Answer: "\u30b3\u30ed\u30ca\u30a6\u30a4\u30eb\u30b9\u306f\u4e16\u754c\u7d4c\u6e08\u306b\u751a\u5927\u306a\u6253\u6483\u3092\u4e0e\u3048\u3001\u591a\u304f\u306e\u4f01\u696d\u304c\u5012\u7523\u3057\u3001\u5931\u696d\u7387\u304c\u6025\u5897\u3057\u307e\u3057\u305f\u3002\u6559\u80b2\u306b\u3064\u3044\u3066\u306f\u3001\u9060\u9694\u5b66\u7fd2\u3078\u306e\u79fb\u884c\u304c\u9032\u307f\u3001\u30c7\u30b8\u30bf\u30eb\u683c\u5dee\u304c\u6d6e\u304d\u5f6b\u308a\u306b\u306a\u308a\u307e\u3057\u305f\u304c\u3001\u65b0\u3057\u3044\u6559\u80b2\u30c6\u30af\u30ce\u30ed\u30b8\u30fc\u306e\u63a1\u7528\u3082\u52a0\u901f\u3057\u307e\u3057\u305f\u3002"
Aspects_Expected: "\u7d4c\u6e08\u3001\u6559\u80b2\u30b7\u30b9\u30c6\u30e0\u3001\u533b\u7642\u30a4\u30f3\u30d5\u30e9"
Aspects_Provided: "\u4e16\u754c\u7d4c\u6e08\u3001\u4f01\u696d\u5012\u7523\u3001\u5931\u696d\u7387\u3001\u9060\u9694\u5b66\u7fd2\u3001\u30c7\u30b8\u30bf\u30eb\u683c\u5dee\u3001\u6559\u80b2\u30c6\u30af\u30ce\u30ed\u30b8\u30fc"
Think: "\u8cea\u554f\u3067\u306f\u660e\u793a\u7684\u306b\u30b3\u30ed\u30ca\u30a6\u30a4\u30eb\u30b9\u306e\u5f71\u97ff\u306e\u4e09\u3064\u306e\u5074\u9762\u306b\u3064\u3044\u3066\u5c0b\u306d\u3066\u3044\u307e\u3059\uff1a\u7d4c\u6e08\u3001\u6559\u80b2\u30b7\u30b9\u30c6\u30e0\u3001\u533b\u7642\u30a4\u30f3\u30d5\u30e9\u3067\u3059\u3002\u56de\u7b54\u306f\u7d4c\u6e08\uff08\u300c\u4e16\u754c\u7d4c\u6e08\u300d\u300c\u4f01\u696d\u5012\u7523\u300d\u300c\u5931\u696d\u7387\u300d\u306b\u3064\u3044\u3066\uff09\u3068\u6559\u80b2\u30b7\u30b9\u30c6\u30e0\uff08\u300c\u9060\u9694\u5b66\u7fd2\u300d\u300c\u30c7\u30b8\u30bf\u30eb\u683c\u5dee\u300d\u300c\u6559\u80b2\u30c6\u30af\u30ce\u30ed\u30b8\u30fc\u300d\u306b\u3064\u3044\u3066\uff09\u306b\u5bfe\u5fdc\u3057\u3066\u3044\u307e\u3059\u304c\u3001\u8cea\u554f\u3067\u660e\u793a\u7684\u306b\u6c42\u3081\u3089\u308c\u3066\u3044\u305f\u533b\u7642\u30a4\u30f3\u30d5\u30e9\u3078\u306e\u5f71\u97ff\u306b\u3064\u3044\u3066\u306e\u8b70\u8ad6\u304c\u5b8c\u5168\u306b\u7701\u7565\u3055\u308c\u3066\u3044\u307e\u3059\u3002"
Pass: false

Question: "\u8bf7\u89e3\u91ca\u4eba\u5de5\u667a\u80fd\u5728\u533b\u7597\u8bca\u65ad\u3001\u81ea\u52a8\u9a7e\u9a76\u548c\u5ba2\u6237\u670d\u52a1\u65b9\u9762\u7684\u5e94\u7528\u3002"
Answer: "\u5728\u533b\u7597\u9886\u57df\uff0cAI\u7b97\u6cd5\u53ef\u4ee5\u5206\u6790\u533b\u5b66\u5f71\u50cf\u4ee5\u68c0\u6d4b\u764c\u75c7\u548c\u5176\u4ed6\u75be\u75c5\uff0c\u51c6\u786e\u7387\u6709\u65f6\u751a\u81f3\u8d85\u8fc7\u4eba\u7c7b\u4e13\u5bb6\u3002\u81ea\u52a8\u9a7e\u9a76\u6280\u672f\u5229\u7528\u673a\u5668\u5b66\u4e60\u5904\u7406\u6765\u81ea\u96f7\u8fbe\u3001\u6fc0\u5149\u96f7\u8fbe\u548c\u6444\u50cf\u5934\u7684\u6570\u636e\uff0c\u5b9e\u65f6\u505a\u51fa\u9a7e\u9a76\u51b3\u7b56\u3002\u5728\u5ba2\u6237\u670d\u52a1\u65b9\u9762\uff0c\u804a\u5929\u673a\u5668\u4eba\u548c\u667a\u80fd\u52a9\u624b\u80fd\u591f\u5904\u7406\u5e38\u89c1\u95ee\u9898\uff0c\u5206\u7c7b\u5ba2\u6237\u67e5\u8be2\uff0c\u5e76\u5728\u5fc5\u8981\u65f6\u5c06\u590d\u6742\u95ee\u9898\u8f6c\u7ed9\u4eba\u5de5\u4ee3\u8868\u3002"
Aspects_Expected: "\u533b\u7597\u8bca\u65ad\u3001\u81ea\u52a8\u9a7e\u9a76\u3001\u5ba2\u6237\u670d\u52a1"
Aspects_Provided: "\u533b\u5b66\u5f71\u50cf\u5206\u6790\u3001\u764c\u75c7\u68c0\u6d4b\u3001\u96f7\u8fbe\u6570\u636e\u5904\u7406\u3001\u6fc0\u5149\u96f7\u8fbe\u6570\u636e\u5904\u7406\u3001\u6444\u50cf\u5934\u6570\u636e\u5904\u7406\u3001\u5b9e\u65f6\u9a7e\u9a76\u51b3\u7b56\u3001\u804a\u5929\u673a\u5668\u4eba\u3001\u667a\u80fd\u52a9\u624b\u3001\u5ba2\u6237\u67e5\u8be2\u5206\u7c7b"
Think: "\u95ee\u9898\u660e\u786e\u8981\u6c42\u89e3\u91ca\u4eba\u5de5\u667a\u80fd\u5728\u4e09\u4e2a\u9886\u57df\u7684\u5e94\u7528\uff1a\u533b\u7597\u8bca\u65ad\u3001\u81ea\u52a8\u9a7e\u9a76\u548c\u5ba2\u6237\u670d\u52a1\u3002\u56de\u7b54\u867d\u7136\u4f7f\u7528\u4e86\u4e0d\u540c\u7684\u672f\u8bed\uff0c\u4f46\u6db5\u76d6\u4e86\u6240\u6709\u4e09\u4e2a\u65b9\u9762\uff1a\u533b\u7597\u8bca\u65ad\uff08\u8ba8\u8bba\u4e86\u2018\u533b\u5b66\u5f71\u50cf\u5206\u6790\u2019\u548c\u2018\u764c\u75c7\u68c0\u6d4b\u2019\uff09\uff0c\u81ea\u52a8\u9a7e\u9a76\uff08\u5305\u62ec\u2018\u96f7\u8fbe\u6570\u636e\u5904\u7406\u2019\u3001\u2018\u6fc0\u5149\u96f7\u8fbe\u6570\u636e\u5904\u7406\u2019\u3001\u2018\u6444\u50cf\u5934\u6570\u636e\u5904\u7406\u2019\u548c\u2018\u5b9e\u65f6\u9a7e\u9a76\u51b3\u7b56\u2019\uff09\uff0c\u4ee5\u53ca\u5ba2\u6237\u670d\u52a1\uff08\u63d0\u5230\u4e86\u2018\u804a\u5929\u673a\u5668\u4eba\u2019\u3001\u2018\u667a\u80fd\u52a9\u624b\u2019\u548c\u2018\u5ba2\u6237\u67e5\u8be2\u5206\u7c7b\u2019\uff09\u3002\u5c3d\u7ba1\u4f7f\u7528\u4e86\u4e0d\u540c\u7684\u8868\u8ff0\uff0c\u4f46\u6240\u6709\u660e\u786e\u63d0\u53ca\u7684\u65b9\u9762\u90fd\u5f97\u5230\u4e86\u5168\u9762\u8986\u76d6\u3002"
Pass: true

Question: "Comment les changements climatiques affectent-ils la production agricole, les \u00e9cosyst\u00e8mes marins et la sant\u00e9 publique dans les r\u00e9gions c\u00f4ti\u00e8res?"
Answer: "Les variations de temp\u00e9rature et de pr\u00e9cipitations modifient les cycles de croissance des cultures et la distribution des ravageurs agricoles, n\u00e9cessitant des adaptations dans les pratiques de culture. Dans les oc\u00e9ans, l'acidification et le r\u00e9chauffement des eaux entra\u00eenent le blanchissement des coraux et la migration des esp\u00e8ces marines vers des latitudes plus froides, perturbant les cha\u00eenes alimentaires existantes."
Aspects_Expected: "production agricole, \u00e9cosyst\u00e8mes marins, sant\u00e9 publique"
Aspects_Provided: "cycles de croissance, distribution des ravageurs, adaptations des pratiques de culture, acidification des oc\u00e9ans, r\u00e9chauffement des eaux, blanchissement des coraux, migration des esp\u00e8ces marines, perturbation des cha\u00eenes alimentaires"
Think: "La question demande explicitement les effets du changement climatique sur trois aspects: la production agricole, les \u00e9cosyst\u00e8mes marins et la sant\u00e9 publique dans les r\u00e9gions c\u00f4ti\u00e8res. La r\u00e9ponse aborde la production agricole (en discutant des 'cycles de croissance', de la 'distribution des ravageurs' et des 'adaptations des pratiques de culture') et les \u00e9cosyst\u00e8mes marins (en couvrant 'l'acidification des oc\u00e9ans', le 'r\u00e9chauffement des eaux', le 'blanchissement des coraux', la 'migration des esp\u00e8ces marines' et la 'perturbation des cha\u00eenes alimentaires'). Cependant, elle omet compl\u00e8tement toute discussion sur les effets sur la sant\u00e9 publique dans les r\u00e9gions c\u00f4ti\u00e8res, qui \u00e9tait explicitement demand\u00e9e dans la question."
Pass: false
</examples>
"""

    user = f"""
Question: {question}
Answer: {answer}

Please look at my answer and think.
"""

    return system, user


def _get_plurality_prompt(question: str, answer: str) -> tuple[str, str]:
    """Build the plurality evaluation prompt."""
    system = """You are an evaluator that analyzes if answers provide the appropriate number of items requested in the question.

<rules>
Question Type Reference Table

| Question Type | Expected Items | Evaluation Rules |
|---------------|----------------|------------------|
| Explicit Count | Exact match to number specified | Provide exactly the requested number of distinct, non-redundant items relevant to the query. |
| Numeric Range | Any number within specified range | Ensure count falls within given range with distinct, non-redundant items. For "at least N" queries, meet minimum threshold. |
| Implied Multiple | \u2265 2 | Provide multiple items (typically 2-4 unless context suggests more) with balanced detail and importance. |
| "Few" | 2-4 | Offer 2-4 substantive items prioritizing quality over quantity. |
| "Several" | 3-7 | Include 3-7 items with comprehensive yet focused coverage, each with brief explanation. |
| "Many" | 7+ | Present 7+ items demonstrating breadth, with concise descriptions per item. |
| "Most important" | Top 3-5 by relevance | Prioritize by importance, explain ranking criteria, and order items by significance. |
| "Top N" | Exactly N, ranked | Provide exactly N items ordered by importance/relevance with clear ranking criteria. |
| "Pros and Cons" | \u2265 2 of each category | Present balanced perspectives with at least 2 items per category addressing different aspects. |
| "Compare X and Y" | \u2265 3 comparison points | Address at least 3 distinct comparison dimensions with balanced treatment covering major differences/similarities. |
| "Steps" or "Process" | All essential steps | Include all critical steps in logical order without missing dependencies. |
| "Examples" | \u2265 3 unless specified | Provide at least 3 diverse, representative, concrete examples unless count specified. |
| "Comprehensive" | 10+ | Deliver extensive coverage (10+ items) across major categories/subcategories demonstrating domain expertise. |
| "Brief" or "Quick" | 1-3 | Present concise content (1-3 items) focusing on most important elements described efficiently. |
| "Complete" | All relevant items | Provide exhaustive coverage within reasonable scope without major omissions, using categorization if needed. |
| "Thorough" | 7-10 | Offer detailed coverage addressing main topics and subtopics with both breadth and depth. |
| "Overview" | 3-5 | Cover main concepts/aspects with balanced coverage focused on fundamental understanding. |
| "Summary" | 3-5 key points | Distill essential information capturing main takeaways concisely yet comprehensively. |
| "Main" or "Key" | 3-7 | Focus on most significant elements fundamental to understanding, covering distinct aspects. |
| "Essential" | 3-7 | Include only critical, necessary items without peripheral or optional elements. |
| "Basic" | 2-5 | Present foundational concepts accessible to beginners focusing on core principles. |
| "Detailed" | 5-10 with elaboration | Provide in-depth coverage with explanations beyond listing, including specific information and nuance. |
| "Common" | 4-8 most frequent | Focus on typical or prevalent items, ordered by frequency when possible, that are widely recognized. |
| "Primary" | 2-5 most important | Focus on dominant factors with explanation of their primacy and outsized impact. |
| "Secondary" | 3-7 supporting items | Present important but not critical items that complement primary factors and provide additional context. |
| Unspecified Analysis | 3-5 key points | Default to 3-5 main points covering primary aspects with balanced breadth and depth. |
</rules>
"""

    user = f"""
Question: {question}
Answer: {answer}

Please look at my answer and think.
"""

    return system, user


def _get_question_evaluation_prompt(question: str) -> tuple[str, str]:
    """Build the question-type evaluation prompt."""
    system = """You are an evaluator that determines if a question requires definitive, freshness, plurality, and/or completeness checks.

<evaluation_types>
definitive - Checks if the question requires a definitive answer or if uncertainty is acceptable (open-ended, speculative, discussion-based)
freshness - Checks if the question is time-sensitive or requires very recent information
plurality - Checks if the question asks for multiple items, examples, or a specific count or enumeration
completeness - Checks if the question explicitly mentions multiple named elements that all need to be addressed
</evaluation_types>

<rules>
1. Definitive Evaluation:
   - Required for ALMOST ALL questions - assume by default that definitive evaluation is needed
   - Not required ONLY for questions that are genuinely impossible to evaluate definitively
   - Examples of impossible questions: paradoxes, questions beyond all possible knowledge
   - Even subjective-seeming questions can be evaluated definitively based on evidence
   - Future scenarios can be evaluated definitively based on current trends and information
   - Look for cases where the question is inherently unanswerable by any possible means

2. Freshness Evaluation:
   - Required for questions about current state, recent events, or time-sensitive information
   - Required for: prices, versions, leadership positions, status updates
   - Look for terms: "current", "latest", "recent", "now", "today", "new"
   - Consider company positions, product versions, market data time-sensitive

3. Plurality Evaluation:
   - ONLY apply when completeness check is NOT triggered
   - Required when question asks for multiple examples, items, or specific counts
   - Check for: numbers ("5 examples"), list requests ("list the ways"), enumeration requests
   - Look for: "examples", "list", "enumerate", "ways to", "methods for", "several"
   - Focus on requests for QUANTITY of items or examples

4. Completeness Evaluation:
   - Takes precedence over plurality check - if completeness applies, set plurality to false
   - Required when question EXPLICITLY mentions multiple named elements that all need to be addressed
   - This includes:
     * Named aspects or dimensions: "economic, social, and environmental factors"
     * Named entities: "Apple, Microsoft, and Google", "Biden and Trump"
     * Named products: "iPhone 15 and Samsung Galaxy S24"
     * Named locations: "New York, Paris, and Tokyo"
     * Named time periods: "Renaissance and Industrial Revolution"
   - Look for explicitly named elements separated by commas, "and", "or", bullets
   - Example patterns: "comparing X and Y", "differences between A, B, and C", "both P and Q"
   - DO NOT trigger for elements that aren't specifically named
</rules>

<examples>
<example-1>
\u8c01\u53d1\u660e\u4e86\u5fae\u79ef\u5206\uff1f\u725b\u987f\u548c\u83b1\u5e03\u5c3c\u5179\u5404\u81ea\u7684\u8d21\u732e\u662f\u4ec0\u4e48\uff1f
<think>
\u8fd9\u662f\u5173\u4e8e\u5fae\u79ef\u5206\u5386\u53f2\u7684\u95ee\u9898\uff0c\u4e0d\u6d89\u53ca\u9700\u8981\u6700\u65b0\u4fe1\u606f\u7684\u5185\u5bb9\u3002\u95ee\u9898\u660e\u786e\u63d0\u5230\u4e86\u725b\u987f\u548c\u83b1\u5e03\u5c3c\u5179\u4e24\u4f4d\u6570\u5b66\u5bb6\uff0c\u8981\u6c42\u5206\u6790\u4ed6\u4eec\u5404\u81ea\u7684\u8d21\u732e\uff0c\u6240\u4ee5\u9700\u8981\u5168\u9762\u8bc4\u4f30\u8fd9\u4e24\u4e2a\u7279\u5b9a\u7684\u65b9\u9762\u3002\u8fd9\u4e2a\u95ee\u9898\u6d89\u53ca\u5386\u53f2\u4e8b\u5b9e\uff0c\u6709\u660e\u786e\u7684\u5b66\u672f\u7814\u7a76\u53ef\u4ee5\u53c2\u8003\uff0c\u56e0\u6b64\u9700\u8981\u786e\u5b9a\u6027\u8bc4\u4f30\u3002
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-1>

<example-2>
fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw \ud83d\ude2d
<think>
This is a mathematical question about eigenvalues which doesn't change over time, so no need for recent information. A 4x4 matrix has multiple eigenvalues, so this requires identifying several distinct values. This is a pure mathematics problem with precise, verifiable solutions that can be definitively evaluated. The question asks for calculation of eigenvalues only, not addressing multiple distinct topics.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-2>

<example-3>
Quelles sont les principales diff\u00e9rences entre le romantisme et le r\u00e9alisme dans la litt\u00e9rature du 19\u00e8me si\u00e8cle?
<think>
C'est une question sur l'histoire litt\u00e9raire, donc aucun besoin d'informations r\u00e9centes. La question mentionne sp\u00e9cifiquement deux mouvements: le romantisme et le r\u00e9alisme. Je dois \u00e9valuer ces deux \u00e9l\u00e9ments nomm\u00e9s, donc l'exhaustivit\u00e9 est importante ici. Cette question porte sur des concepts litt\u00e9raires \u00e9tablis avec des caract\u00e9ristiques document\u00e9es, donc une \u00e9valuation d\u00e9finitive est possible. La question ne demande pas une liste ou \u00e9num\u00e9ration multiple au-del\u00e0 des deux mouvements sp\u00e9cifi\u00e9s.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-3>

<example-4>
Shakespeare \u306e\u6700\u3082\u6709\u540d\u306a\u60b2\u5287\u30925\u3064\u6319\u3052\u3001\u7c21\u5358\u306b\u3042\u3089\u3059\u3058\u3092\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002
<think>
\u30b7\u30a7\u30a4\u30af\u30b9\u30d4\u30a2\u306e\u60b2\u5287\u306b\u3064\u3044\u3066\u306e\u8cea\u554f\u3067\u3042\u308a\u3001\u6642\u4e8b\u7684\u306a\u60c5\u5831\u306f\u4e0d\u8981\u3002\u300c5\u3064\u6319\u3052\u300d\u3068\u3044\u3046\u6307\u5b9a\u304c\u3042\u308b\u305f\u3081\u3001\u8907\u6570\u306e\u9805\u76ee\u304c\u6c42\u3081\u3089\u308c\u3066\u3044\u308b\u3002\u300c\u6700\u3082\u6709\u540d\u306a\u300d\u3068\u3044\u3046\u57fa\u6e96\u306f\u5b66\u8853\u7684\u30b3\u30f3\u30bb\u30f3\u30b5\u30b9\u3084\u6587\u5316\u7684\u91cd\u8981\u6027\u306b\u57fa\u3065\u3044\u3066\u5224\u65ad\u3067\u304d\u308b\u305f\u3081\u3001\u78ba\u5b9a\u7684\u306a\u8a55\u4fa1\u304c\u53ef\u80fd\u3002\u7279\u5b9a\u306e\u4f5c\u54c1\u3092\u5206\u6790\u3059\u308b\u3088\u3046\u6c42\u3081\u3089\u308c\u3066\u3044\u308b\u308f\u3051\u3067\u306f\u306a\u304f\u3001\u8907\u6570\u306e\u4f5c\u54c1\u3092\u5217\u6319\u3059\u308b\u3053\u3068\u304c\u4e3b\u306a\u8981\u4ef6\u3002
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-4>

<example-5>
What are the current interest rates for mortgage loans from Bank of America, Wells Fargo, and Chase Bank in the US?
<think>
This question asks about "current" interest rates, so it clearly requires up-to-date information. The query specifically names three banks: Bank of America, Wells Fargo, and Chase Bank. Each of these named entities must be addressed, making completeness necessary. This question seeks factual financial data that can be objectively verified, so definitive evaluation is needed. The question isn't asking for multiple types of information beyond the specified banks.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-5>

<example-6>
2025\u5e74\u306b\u6ce8\u76ee\u3059\u3079\u304d\u4eba\u5de5\u77e5\u80fd\u306e3\u3064\u306e\u30c8\u30ec\u30f3\u30c9\u306f\u4f55\u3067\u3059\u304b\uff1f
<think>
\u3053\u308c\u306f\u5c06\u6765\u306eAI\u30c8\u30ec\u30f3\u30c9\u306b\u95a2\u3059\u308b\u8cea\u554f\u306a\u306e\u3067\u3001\u6700\u65b0\u306e\u60c5\u5831\u304c\u5fc5\u8981\u3002\u300c3\u3064\u306e\u300d\u3068\u3044\u3046\u660e\u78ba\u306a\u6570\u3092\u6307\u5b9a\u3057\u3066\u3044\u308b\u305f\u3081\u3001\u8907\u6570\u306e\u9805\u76ee\u304c\u6c42\u3081\u3089\u308c\u3066\u3044\u308b\u3002\u672a\u6765\u306e\u4e88\u6e2c\u3060\u304c\u3001\u73fe\u5728\u306eAI\u958b\u767a\u52d5\u5411\u3084\u7814\u7a76\u304b\u3089\u63a8\u6e2c\u3067\u304d\u308b\u7bc4\u56f2\u3067\u3042\u308a\u3001\u78ba\u5b9a\u7684\u306a\u8a55\u4fa1\u304c\u53ef\u80fd\u3002\u7279\u5b9a\u306e\u5074\u9762\u306b\u3064\u3044\u3066\u8a00\u53ca\u3057\u3066\u304a\u3089\u305a\u3001\u30c8\u30ec\u30f3\u30c9\u306e\u5217\u6319\u304c\u4e3b\u306a\u8981\u4ef6\u306a\u306e\u3067\u3001\u8907\u6570\u6027\u306e\u307b\u3046\u304c\u91cd\u8981\u3002
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-6>

<example-7>
Was sind die besten Strategien f\u00fcr nachhaltiges Investieren in der heutigen Wirtschaft?
<think>
Diese Frage bezieht sich auf die "heutige Wirtschaft", daher sind aktuelle Informationen erforderlich. "Strategien" steht im Plural, was auf die Notwendigkeit mehrerer Beispiele hindeutet. Obwohl "beste" subjektiv klingen mag, l\u00e4sst sich die Frage anhand von Rendite-Daten, Risikobewertungen und Nachhaltigkeitskriterien definitiv evaluieren. Es werden keine spezifischen Aspekte genannt, die alle behandelt werden m\u00fcssen - der Fokus liegt auf der Vielfalt der Strategien.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-7>

<example-8>
\u8bf7\u89e3\u91ca\u8d64\u58c1\u4e4b\u6218\u7684\u5386\u53f2\u80cc\u666f\u3001\u4e3b\u8981\u53c2\u4e0e\u8005\u4ee5\u53ca\u6218\u7565\u610f\u4e49\uff0c\u8fd9\u5bf9\u4e2d\u56fd\u5386\u53f2\u4ea7\u751f\u4e86\u4ec0\u4e48\u5f71\u54cd\uff1f
<think>
\u8fd9\u662f\u5173\u4e8e\u5386\u53f2\u4e8b\u4ef6\u7684\u95ee\u9898\uff0c\u4e0d\u9700\u8981\u6700\u65b0\u4fe1\u606f\u3002\u95ee\u9898\u660e\u786e\u5217\u51fa\u4e86\u591a\u4e2a\u9700\u8981\u8bc4\u4f30\u7684\u65b9\u9762\uff1a\u5386\u53f2\u80cc\u666f\u3001\u4e3b\u8981\u53c2\u4e0e\u8005\u3001\u6218\u7565\u610f\u4e49\u548c\u5386\u53f2\u5f71\u54cd\u3002\u8fd9\u4e9b\u90fd\u662f\u7279\u5b9a\u7684\u5386\u53f2\u8bfe\u9898\uff0c\u9700\u8981\u5b8c\u6574\u5730\u8986\u76d6\u3002\u5173\u4e8e\u5386\u53f2\u4e8b\u4ef6\u7684\u95ee\u9898\u6709\u660e\u786e\u7684\u53f2\u6599\u8bb0\u8f7d\u548c\u5b66\u672f\u7814\u7a76\uff0c\u53ef\u4ee5\u8fdb\u884c\u786e\u5b9a\u6027\u8bc4\u4f30\u3002
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-8>

<example-9>
How might artificial intelligence transform the healthcare industry over the next decade?
<think>
This question asks about future developments that require recent AI and healthcare information to properly evaluate. The query is about transformations (plural) suggesting multiple changes should be identified. While it concerns future trends, the patterns can be evaluated definitively based on current research, development pipelines, and industry analyses. No specific aspects of healthcare are mentioned that must all be addressed, so plurality is more relevant than completeness.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-9>

<example-10>
What do you think is the best approach to learning a new language as an adult?
<think>
This question about language learning does not require time-sensitive information as fundamental language acquisition principles remain relatively stable. The phrasing suggests looking for multiple elements or methods that constitute an effective approach, indicating plurality. Though the question uses subjective language ("what do you think"), it can be evaluated definitively using linguistic research and proven methodologies. No specific language learning aspects are explicitly named that must all be covered.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-10>

<example-11>
If a tree falls in a forest with absolutely no observers, instruments, or any possible way to detect it, does it make a sound?
<think>
This is a classic philosophical paradox that is inherently unanswerable in a definitive way. The question deliberately constructs a scenario that removes all possible means of verification, making it logically impossible to evaluate. This kind of question represents one of the rare cases where a definitive evaluation is truly impossible. The question doesn't involve recent events, doesn't request multiple items, and doesn't specify multiple elements that must be addressed.
</think>
<output>
"needsDefinitive": false,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": false,
</output>
</example-11>
</examples>

"""

    user = f"""
{question}
<think>"""

    return system, user


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def evaluate_question(
    question: str,
    generate_fn: GenerateFn,
    schema_gen: SchemaGenFn,
) -> list[str]:
    """Determine which evaluation types apply to this question.

    Args:
        question: The user's question.
        generate_fn: Async callable ``(schema, system, prompt) -> BaseModel``.
        schema_gen: Callable that returns a Pydantic model class for a given
            evaluation type string. Called with ``"question_evaluate"`` here.

    Returns:
        Ordered list of evaluation type strings to run (e.g.
        ``["definitive", "freshness", "plurality"]``).
    """
    try:
        prompt_system, prompt_user = _get_question_evaluation_prompt(question)
        schema = schema_gen("question_evaluate")
        result = await generate_fn(schema, prompt_system, prompt_user)

        types: list[str] = []
        if getattr(result, "needsDefinitive", False) or getattr(result, "needs_definitive", False):
            types.append("definitive")
        if getattr(result, "needsFreshness", False) or getattr(result, "needs_freshness", False):
            types.append("freshness")
        if getattr(result, "needsPlurality", False) or getattr(result, "needs_plurality", False):
            types.append("plurality")
        if getattr(result, "needsCompleteness", False) or getattr(result, "needs_completeness", False):
            types.append("completeness")

        logger.info("evaluate_question: question=%s types=%s", question[:80], types)
        return types

    except Exception:
        logger.exception("Error in question evaluation")
        # Default to no checks on error
        return []


async def evaluate_answer(
    question: str,
    answer_action: Any,  # AnswerAction
    eval_types: list[str],
    generate_fn: GenerateFn,
    schema_gen: SchemaGenFn,
    knowledge: list[Any] | None = None,
) -> dict:
    """Run sequential evaluation -- fail fast on first failure.

    Args:
        question: The user's question.
        answer_action: An ``AnswerAction`` (or compatible) with an ``.answer``
            attribute.
        eval_types: Ordered list of evaluation types to run (from
            :func:`evaluate_question`).
        generate_fn: Async callable ``(schema, system, prompt) -> BaseModel``.
        schema_gen: Callable that returns a Pydantic model class for a given
            evaluation type string.
        knowledge: Optional list of ``KnowledgeItem`` objects used by the
            strict evaluator.

    Returns:
        Dict with at least ``"pass"`` (bool), ``"think"`` (str), and
        ``"type"`` (str) keys, mirroring ``EvaluationResponse``.
    """
    result: BaseModel | None = None

    for eval_type in eval_types:
        prompt: tuple[str, str] | None = None

        if eval_type == "definitive":
            prompt = _get_definitive_prompt(question, answer_action.answer)
        elif eval_type == "freshness":
            current_time = datetime.now(timezone.utc).isoformat()
            prompt = _get_freshness_prompt(question, answer_action, current_time)
        elif eval_type == "plurality":
            prompt = _get_plurality_prompt(question, answer_action.answer)
        elif eval_type == "completeness":
            prompt = _get_completeness_prompt(question, answer_action.answer)
        elif eval_type == "strict":
            prompt = _get_reject_all_answers_prompt(
                question, answer_action, knowledge or []
            )
        else:
            logger.error("Unknown evaluation type: %s", eval_type)

        if prompt is not None:
            try:
                schema = schema_gen(eval_type)
                result = await generate_fn(schema, prompt[0], prompt[1])
            except Exception:
                logger.exception("Error performing %s evaluation", eval_type)
                return {
                    "pass": False,
                    "think": (
                        f"Error {eval_type} immediately return false, "
                        "probably due to bad prompt?"
                    ),
                    "type": eval_type,
                }

            # Fail-fast: if this evaluation did not pass, return immediately
            passed = getattr(result, "passed", None)
            if passed is None:
                # Also try the literal "pass" key via dict access for
                # models that use alias
                passed = getattr(result, "pass", None)
            if passed is None and hasattr(result, "model_dump"):
                dump = result.model_dump(by_alias=True)
                passed = dump.get("pass", dump.get("passed"))

            if not passed:
                return _result_to_dict(result, eval_type)

    # All evaluations passed -- return the last result
    if result is not None:
        return _result_to_dict(result, eval_types[-1] if eval_types else "unknown")
    # No evaluations were run
    return {"pass": True, "think": "No evaluations required", "type": "none"}


def _result_to_dict(result: BaseModel | None, eval_type: str) -> dict:
    """Convert a Pydantic evaluation result to a plain dict."""
    if result is None:
        return {"pass": False, "think": "", "type": eval_type}
    if hasattr(result, "model_dump"):
        d = result.model_dump(by_alias=True)
    else:
        d = dict(result)  # type: ignore[arg-type]
    # Ensure 'type' is present
    if "type" not in d:
        d["type"] = eval_type
    return d
