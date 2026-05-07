#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/channel.h"
#include "common/result.h"
#include "common/types.h"

namespace ccinfer {
namespace engine {

class Executor {
public:
    virtual ~Executor() = default;

    virtual Result<void> init(const std::string& model_path) = 0;
    virtual void shutdown() = 0;

    virtual void enqueue_create_sequence(std::vector<int32_t> prompt_tokens, int max_context_len,
                                         std::shared_ptr<SeqIdChannel> chan) = 0;
    virtual void enqueue_release_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) = 0;
    virtual void enqueue_abort_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) = 0;
    virtual void enqueue_execute_batch(ScheduledBatch batch,
                                       std::shared_ptr<BatchChannel> chan) = 0;

    virtual EngineCapacity capacity() const = 0;
};

}  // namespace engine
}  // namespace ccinfer
