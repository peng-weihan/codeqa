"""
Run script for analyzing SWE-bench_Lite statements and categorizing generated questions.
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SWE-bench_Lite statements and categorize generated questions."
    )
    parser.add_argument(
        "--use-api", 
        action="store_true",
        help="Use OpenAI API for question generation"
    )
    parser.add_argument(
        "--api-key", 
        type=str,
        help="OpenAI API key (optional, can also use OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=50,
        help="Limit the number of examples to process (default: 50)"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to use (default: test)"
    )
    # LLM model configuration parameters
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-3.5-turbo",
        help="Model to use for question generation (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for text generation (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=500,
        help="Maximum number of tokens to generate (default: 500)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=60,
        help="API call timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3,
        help="Maximum number of API call retries (default: 3)"
    )
    parser.add_argument(
        "--use-azure", 
        action="store_true",
        help="Use Azure OpenAI API instead of standard OpenAI API"
    )
    parser.add_argument(
        "--azure-endpoint", 
        type=str,
        help="Azure OpenAI API endpoint (required if --use-azure is specified)"
    )
    parser.add_argument(
        "--azure-api-version", 
        type=str,
        default="2023-05-15",
        help="Azure OpenAI API version (default: 2023-05-15)"
    )
    parser.add_argument(
        "--azure-deployment", 
        type=str,
        help="Azure OpenAI deployment name (required if --use-azure is specified)"
    )
    
    args = parser.parse_args()
    
    # Set OpenAI API key if provided
    if args.use_api:
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            print("Error: OpenAI API key is required when using --use-api.")
            print("Provide it via --api-key or set OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    # Choose which script to run based on whether to use API
    if args.use_api:
        # Import here to avoid loading API-dependent modules if not needed
        import llm_question_generator
        from llm_models_config import ChatOpenAIConfig, AzureChatOpenAIConfig
        
        # Set the module variables
        llm_question_generator.total_count_limit = args.limit
        llm_question_generator.split_name = args.split
        
        # Configure the model based on command line arguments
        if args.use_azure:
            if not args.azure_endpoint or not args.azure_deployment:
                print("Error: When using Azure OpenAI, both --azure-endpoint and --azure-deployment are required.")
                sys.exit(1)
            
            # Create Azure OpenAI configuration
            config = AzureChatOpenAIConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                azure_endpoint=args.azure_endpoint,
                openai_api_version=args.azure_api_version,
                deployment_name=args.azure_deployment
            )
        else:
            # Create standard OpenAI configuration
            config = ChatOpenAIConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries
            )
        
        # Set the configuration in the module
        llm_question_generator.default_openai_config = config
        
        # Call the main function
        llm_question_generator.main()
    else:
        # Import here to avoid potential circular imports
        import analyze_statements
        
        # Set the module variables
        analyze_statements.total_count_limit = args.limit
        analyze_statements.split_name = args.split
        
        # Call the main function
        analyze_statements.main()

if __name__ == "__main__":
    main() 