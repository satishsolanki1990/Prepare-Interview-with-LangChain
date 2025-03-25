# utility.py (Refactored and Improved)

# Curated list of common IT skills across domains for resume parsing and JD matching
TECHNICAL_SKILLS = list(set([
    'python', 'java', 'c++', 'r', 'html', 'css', 'javascript', 'react', 'node.js', 'express.js', 'sql',
    'mongodb', 'restful apis', 'git', 'unit testing', 'docker', 'kubernetes', 'jenkins', 'ansible',
    'ci/cd', 'aws', 'azure', 'google cloud', 'infrastructure as code', 'terraform', 'cloud platforms',
    'cloud infrastructure design', 'cloud security', 'cloud services', 'linux', 'windows', 'mac os',
    'network security', 'virtualization',
    'machine learning', 'deep learning', 'data analysis', 'pandas', 'numpy', 'data visualization',
    'data cleaning', 'statistics', 'excel', 'tableau', 'power bi', 'tensorflow', 'pytorch', 'keras',
    'model evaluation', 'regression', 'natural language processing', 'reinforcement learning',
    'neural networks', 'data preprocessing', 'big data technologies', 'hadoop', 'spark', 'etl',
    'etl pipelines', 'data modeling', 'data warehousing', 'algorithm design', 'problem-solving',
    'debugging', 'test plans', 'qa engineering', 'automation testing', 'performance testing',
    'manual testing', 'regression testing', 'selenium', 'junit', 'cucumber', 'software testing',
    'bug tracking', 'security testing', 'penetration testing', 'threat analysis',
    'security governance', 'compliance', 'soc', 'risk management',
    'project management', 'product management', 'agile', 'scrum master', 'team leadership',
    'strategic planning', 'stakeholder management', 'budgeting', 'mobile ui/ux', 'app design',
    'android', 'ios', 'swift', 'kotlin', 'react native', 'flutter', 'unity', 'unreal engine',
    'game physics', '3d modeling', 'multiplayer design', 'ai for games', 'microcontrollers', 
    'circuit design', 'sensors', 'iot', 'business analysis', 'systems analysis']))

# Prompt templates used in AI resume and JD evaluations
PROMPT_TEMPLATES = {
    "tech_hr_review": """
        You are a Technical HR Manager. Assess the resume against the job description:
        - List matched technical skills in four lines.
        - Show JD experience requirements and candidate experience side-by-side in bullet points.
        - Highlight missing skills in four bullet points.
        - Provide a brief 10-word recommendation on whether to apply.
    """,

    "ats_review_simple": """
        You are an ATS expert. Compare resume keywords with job description.
        - Display overall ATS Score (e.g., ATS Score: 85%) as a title.
        - Report matched keyword percentage.
        - Bullet out missing JD keywords only.
        Limit the response to around 40 words.
    """,

    "ats_review_advanced": """
        You are an ATS expert. Evaluate resume fit for a job description.
        - Show ATS Score based on skills/experience match.
        - If score > 60%, provide a short positive comment.
        - If score < 60%, suggest specific improvements.
        - Include percentage of matched skills.
        - Give one 20-word advice bullet based on overall fit.
        - Bullet out missing JD keywords only.
        Response should stay concise (~40 words).
    """,

    "job_description_extraction": """
        You are a Technical HR Analyst. Extract key data from a job description:
        - Job Title
        - Qualifications
        - Required Skills
        - Years/Type of Experience
        - Company Name (if mentioned, else "Not specified")
        Output should be a JSON with keys: job_title, qualifications, skills, experience, company.
        If any value is missing, use "Not specified".
    """
}
