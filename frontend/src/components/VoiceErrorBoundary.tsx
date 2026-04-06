import React from 'react';

type VoiceErrorBoundaryProps = {
  children: React.ReactNode;
};

type VoiceErrorBoundaryState = {
  hasError: boolean;
  message: string;
};

class VoiceErrorBoundary extends React.Component<VoiceErrorBoundaryProps, VoiceErrorBoundaryState> {
  constructor(props: VoiceErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error: Error): VoiceErrorBoundaryState {
    return {
      hasError: true,
      message: error?.message || 'Unknown runtime error',
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('Voice page runtime error:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, message: '' });
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="voice-page">
        <div className="voice-card">
          <h2>语音页面发生错误</h2>
          <p style={{ marginTop: 8, color: 'var(--text-secondary)' }}>
            {this.state.message || 'Runtime error'}
          </p>
          <button style={{ marginTop: 14 }} onClick={this.handleReset}>
            重试加载语音页面
          </button>
        </div>
      </div>
    );
  }
}

export default VoiceErrorBoundary;

