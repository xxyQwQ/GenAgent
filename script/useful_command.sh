# Run Zero-shot Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "zero_shot_agent" --save_path "checkpoint/agent-test/zero-shot-agent"

# Run Few-shot Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "few_shot_agent" --save_path "checkpoint/agent-test/few-shot-agent"

# Run CoT Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "cot_agent" --save_path "checkpoint/agent-test/cot-agent"

# Run RAG Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "rag_agent" --save_path "checkpoint/agent-test/rag-agent"

# Run Gen Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "gen_agent" --save_path "checkpoint/agent-test/gen-agent"

# Run Json Represented Gen Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "json_gen_agent" --save_path "checkpoint/agent-test/json-gen-agent"

# Run List Represented Gen Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "list_gen_agent" --save_path "checkpoint/agent-test/list-gen-agent"

# Run Code Represented Gen Agent
python main.py --query_text "Generate an image of a male basketball player shooting in a gym. The player should be wearing a white jersey and black shorts and looking at the basket. The background should be a gym with a basketball hoop. The result should be a 1024x768 image." --agent_name "code_gen_agent" --save_path "checkpoint/agent-test/code-gen-agent"

# Run Benchmark Evaluation
python inference.py --agent_name "zero_shot_agent" "few_shot_agent" "cot_agent" "rag_agent" "gen_agent"
python evaluation.py --agent_name "zero_shot_agent" "few_shot_agent" "cot_agent" "rag_agent" "gen_agent"

# Run Representation Ablation
python inference.py --agent_name "json_gen_agent" "list_gen_agent" "code_gen_agent" --num_fixes 0
python evaluation.py --agent_name "json_gen_agent" "list_gen_agent" "code_gen_agent"
