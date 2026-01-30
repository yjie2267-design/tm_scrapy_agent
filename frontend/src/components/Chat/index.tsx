import { AgentScopeRuntimeWebUI, IAgentScopeRuntimeWebUIRef, IAgentScopeRuntimeWebUIOptions } from '@agentscope-ai/chat';
import OptionsPanel from './OptionsPanel';
import { useMemo, useRef, useEffect } from 'react';
import sessionApi from './sessionApi';
import { useLocalStorageState } from 'ahooks';
import defaultConfig from './OptionsPanel/defaultConfig';
import Weather from '../Cards/Weather';

export default function () {
  const chatRef = useRef<IAgentScopeRuntimeWebUIRef>(null);
  // @ts-ignore
  window.chatRef = chatRef;

  const [optionsConfig, setOptionsConfig] = useLocalStorageState('agent-scope-runtime-webui-options', {
    defaultValue: defaultConfig,
    listenStorageChange: true,
  });

  // 初始化 localStorage：如果缺少配置，立即设置默认值
  useEffect(() => {
    const currentConfigStr = localStorage.getItem('agent-scope-runtime-webui-options');
    if (!currentConfigStr) {
      console.log('⚠️ localStorage 为空，设置默认配置');
      setOptionsConfig(defaultConfig);
    }
  }, []);

  const options = useMemo(() => {
    const uploadBaseURL = optionsConfig.api?.baseURL.replace('/process', '') || ''; // TODO: 从环境变量中获取
    const rightHeader = <OptionsPanel value={optionsConfig} onChange={(v: typeof optionsConfig) => {
      setOptionsConfig(prev => ({
        ...prev,
        ...v,
      }));
    }} />;

    const result = {
      ...optionsConfig,
      session: {
        multiple: true,
        api: sessionApi,
      },
      theme: {
        ...optionsConfig.theme,
        rightHeader,
      },
      sender: {
        ...optionsConfig.sender,
        attachments: optionsConfig.sender?.attachments ? {

          customRequest(options: any) {
            const file = options.file as File;

            console.log('📤️ Uploading file:', file.name);
            console.log('🌐 Upload URL:', `${uploadBaseURL}/upload`);
            console.log('📦 Base64 length:', file.size, 'bytes');

            // 模拟上传进度
            options.onProgress?.({ percent: 0 });

            // 使用 FileReader 读取文件
            const reader = new FileReader();

            reader.onload = async () => {
              try {
                options.onProgress?.({ percent: 50 });

                const base64 = reader.result as string;
                console.log('✅ File converted to base64, length:', base64.length);

                const uploadUrl = `${uploadBaseURL}/upload`;
                console.log('🚀 Fetching:', uploadUrl);

                const response = await fetch(uploadUrl, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json',
                  },
                  body: JSON.stringify({
                    filename: file.name,
                    file_data: base64,
                  }),
                });

                options.onProgress?.({ percent: 80 });

                console.log('📊 Response status:', response.status, response.ok);

                if (!response.ok) {
                  console.error('❌ Upload failed with status:', response.status);
                  options.onError?.(new Error(`HTTP error: ${response.status}`));
                  return;
                }

                const data = await response.text();
                console.log('📦 Upload response:', data);

                if (data.status === 400 || data.status === 500) {
                  console.error('❌ Server error:', data.error);
                  options.onError?.(new Error(data.error || 'Upload failed'));
                  return;
                }

                options.onProgress?.({ percent: 100 });

                console.log('✅ Upload successful, file_url:', data.file_url);
                options.onSuccess?.({
                  url: data.file_url,
                  file_id: data.file_id,
                });
              } catch (error) {
                console.error('❌ Upload error:', error);
                options.onError?.(error instanceof Error ? error : new Error('Upload failed'));
              }
            };

            reader.onerror = () => {
              console.error('❌ Failed to read file');
              options.onError?.(new Error('Failed to read file'));
            };

            // 读取文件为 base64
            reader.readAsDataURL(file);
          }
        } : undefined,
      },
      customToolRenderConfig: {
        'weather search mock': Weather,
      },
    } as unknown as IAgentScopeRuntimeWebUIOptions;


    return result;
  }, [optionsConfig]);

  return <div style={{ height: '100vh' }}>
    <AgentScopeRuntimeWebUI
      options={options}
    />
  </div>;
}
