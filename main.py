import argparse
from examples.basic_temporal_reasoning import basic_temporal_reasoning_demo
from examples.advanced_causal_analysis import advanced_causal_analysis_demo

def main():
    parser = argparse.ArgumentParser(description="Temporal Reasoning Vision System")
    parser.add_argument('--mode', type=str, choices=['demo', 'train', 'analyze'], default='demo')
    parser.add_argument('--video', type=str, help='Input video file path')
    parser.add_argument('--tasks', nargs='+', help='Reasoning tasks to perform')
    parser.add_argument('--output', type=str, help='Output results file')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Temporal Reasoning Vision System Demo")
        basic_temporal_reasoning_demo()
    
    elif args.mode == 'train':
        print("Running Advanced Training Pipeline")
        advanced_causal_analysis_demo()
    
    elif args.mode == 'analyze':
        if not args.video:
            print("Please provide a video file with --video")
            return
        
        engine = TemporalReasoningEngine()
        results = engine.process_video(args.video, args.tasks)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print("Analysis Results:", results)

if __name__ == "__main__":
    main()