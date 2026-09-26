/*
 * XorWoW price policy for the AuctionHouseBot.
 *
 * The buyer bot pays with gold that does not come from anywhere, and the seller bot creates its items from nothing, so
 * neither may offer a deal that a player can turn into gold. At startup (and on config reload) this works out:
 *
 * Buyer price ceilings, one per item:
 *   - grey items, items above Buyer.MaxQuality and items a vendor sells for gold without a stock limit are never bought
 *   - items with a vendor value: at most Buyer.MaxVendorValueMultiplier(.Rare) times that value
 *   - items without a vendor value: only the ones listed in Buyer.ZeroValueItemPrices (enchanting materials)
 *   - never more than SafetyMargin of the cheapest known way to obtain the item: a vendor at the best reputation
 *     discount (limited stock only if it restocks 10+ an hour), the seller bot's lowest possible price, or a
 *     profession recipe made from such items
 *   - an auction priced at most SafetyMargin of the seller bot's own cheapest live listing of the item may be bought
 *     above the vendor value ceiling (AuctionHouseBot.cpp), within SafetyMargin of the cheapest other way to obtain it
 *     (vendor, recipe) and never above the seller's lowest possible price, so buying from the seller to sell back to
 *     the buyer always loses, however far apart the seller's listings are
 *
 * Loops through the seller bot's items: when crafting, disenchanting, prospecting, milling or opening them pays more
 * (to a vendor, or to the buyer after SafetyMargin) than the input cost, the seller's minimum price for the items in
 * the loop is raised just enough to close it; only when that would take more than MaxSellerFloorRaise are they kept
 * off the auction house, as are items the seller could list below their vendor value. When a loop's input comes from
 * a vendor instead, the buyer's ceilings for its outputs are lowered until the loop loses money.
 */

#include "AuctionHouseBot.h"
#include "DBCStores.h"
#include "DatabaseEnv.h"
#include "Field.h"
#include "Log.h"
#include "ObjectMgr.h"
#include "QueryResult.h"
#include "SpellInfo.h"
#include "SpellMgr.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <limits>
#include <set>
#include <vector>

namespace
{
    // Best vendor reputation discount (exalted)
    constexpr double VendorBestDiscount = 0.80;
    // The buyer pays at most this share of the cheapest known way to obtain an item. With the 5% auction house cut on
    // top, any buy-and-resell loop loses at least 19%
    constexpr double SafetyMargin = 0.85;
    // A vendor with limited stock only counts as a cheap source for loops when it restocks at least this many items
    // an hour, summed over every vendor selling the item
    constexpr double VendorLoopRestockPerHour = 10.0;
    constexpr uint32 MaxLootDepth = 6;
    constexpr uint32 MaxRecipeDepth = 10;
    constexpr uint32 MaxPolicyRounds = 50;
    // A loop is closed by raising the seller's minimum price for its items at most this much; beyond that they are not listed
    constexpr double MaxSellerFloorRaise = 20.0;

    struct LootRow
    {
        uint32 Item;
        int32 Reference;
        float Chance;
        uint8 GroupId;
        uint8 MinCount;
        uint8 MaxCount;
    };

    using LootTable = std::unordered_map<uint32, std::vector<LootRow>>;
    using ItemCounts = std::unordered_map<uint32, double>;

    struct CraftRecipe
    {
        uint32 SpellId;
        uint32 CreatedItem;       // 0 for recipes whose result comes from spell_loot_template only
        double CreatedCount;
        std::vector<std::pair<uint32, uint32>> Reagents;
    };

    enum class CostSource
    {
        Vendor,
        Seller,
        Craft
    };

    struct KnownCost
    {
        double Cost;
        CostSource Source;
        CraftRecipe const* Recipe;
    };

    struct LootProcess
    {
        uint32 SourceItem;        // 0 when the input is a recipe's reagents
        uint32 SourceCount;
        CraftRecipe const* Recipe;
        ItemCounts Yield;
        double Money;
        char const* Name;
    };

    LootTable LoadLootTable(char const* table)
    {
        LootTable result;
        QueryResult rows = WorldDatabase.Query("SELECT Entry, Item, Reference, Chance, GroupId, MinCount, MaxCount FROM {} WHERE QuestRequired = 0 AND (LootMode & 1) <> 0", table);
        if (!rows)
            return result;

        do
        {
            Field* fields = rows->Fetch();
            result[fields[0].Get<uint32>()].push_back({ fields[1].Get<uint32>(), fields[2].Get<int32>(), fields[3].Get<float>(),
                fields[4].Get<uint8>(), fields[5].Get<uint8>(), fields[6].Get<uint8>() });
        } while (rows->NextRow());

        return result;
    }

    // Adds the expected number of each item that one roll of a loot entry yields
    void AddExpectedLoot(LootTable const& table, uint32 entry, double weight, LootTable const& references, ItemCounts& out, uint32 depth = 0)
    {
        if (depth > MaxLootDepth)
            return;

        auto rowsItr = table.find(entry);
        if (rowsItr == table.end())
            return;

        // Rows of a group without a chance share whatever the explicitly chanced rows leave over
        std::unordered_map<uint8, double> groupExplicitChance;
        std::unordered_map<uint8, uint32> groupEqualChanceRows;
        for (LootRow const& row : rowsItr->second)
        {
            if (!row.GroupId)
                continue;
            if (row.Chance > 0)
                groupExplicitChance[row.GroupId] += row.Chance;
            else
                ++groupEqualChanceRows[row.GroupId];
        }

        for (LootRow const& row : rowsItr->second)
        {
            double chance = row.Chance / 100.0;
            if (row.GroupId && row.Chance <= 0)
                chance = std::max(0.0, 100.0 - groupExplicitChance[row.GroupId]) / 100.0 / groupEqualChanceRows[row.GroupId];
            chance = std::min(chance, 1.0);
            if (chance <= 0)
                continue;

            // A reference row is processed MaxCount times; the core reads negative references as positive
            if (row.Reference != 0)
                AddExpectedLoot(references, uint32(std::abs(row.Reference)), weight * chance * std::max<uint8>(row.MaxCount, 1), references, out, depth + 1);
            else
                out[row.Item] += weight * chance * (row.MinCount + row.MaxCount) / 2.0;
        }
    }

    // Profession recipes (primary and secondary professions) that create items
    std::vector<CraftRecipe> LoadProfessionRecipes(std::unordered_set<uint32> const& spellsWithLoot)
    {
        std::unordered_set<uint32> professionSpells;
        for (uint32 i = 0; i < sSkillLineAbilityStore.GetNumRows(); ++i)
        {
            SkillLineAbilityEntry const* ability = sSkillLineAbilityStore.LookupEntry(i);
            if (!ability)
                continue;
            SkillLineEntry const* skillLine = sSkillLineStore.LookupEntry(ability->SkillLine);
            if (skillLine && (skillLine->categoryId == SKILL_CATEGORY_PROFESSION || skillLine->categoryId == SKILL_CATEGORY_SECONDARY))
                professionSpells.insert(ability->Spell);
        }

        std::vector<CraftRecipe> recipes;
        for (uint32 spellId : professionSpells)
        {
            SpellInfo const* spellInfo = sSpellMgr->GetSpellInfo(spellId);
            if (!spellInfo)
                continue;

            CraftRecipe recipe{ spellId, 0, 0.0, {} };
            for (uint32 i = 0; i < MAX_SPELL_REAGENTS; ++i)
                if (spellInfo->Reagent[i] > 0 && spellInfo->ReagentCount[i] > 0)
                    recipe.Reagents.push_back({ uint32(spellInfo->Reagent[i]), spellInfo->ReagentCount[i] });

            for (SpellEffectInfo const& effect : spellInfo->Effects)
            {
                if ((effect.Effect == SPELL_EFFECT_CREATE_ITEM || effect.Effect == SPELL_EFFECT_CREATE_ITEM_2) && effect.ItemType)
                {
                    recipe.CreatedItem = effect.ItemType;
                    // Largest possible stack, so the cost per created item is never overestimated
                    recipe.CreatedCount = std::max(1, effect.BasePoints + std::max(effect.DieSides, 1));
                    break;
                }
            }

            if (recipe.CreatedItem || spellsWithLoot.count(spellId))
                recipes.push_back(std::move(recipe));
        }

        return recipes;
    }
}

uint64 AuctionHouseBot::GetSellerMinimumPricePerItem(ItemTemplate const* itemProto)
{
    uint64 bidPrice = 0;
    uint64 buyoutPrice = 0;
    CalculateItemValue(itemProto, bidPrice, buyoutPrice, true);
    return std::min(bidPrice, buyoutPrice);
}

void AuctionHouseBot::BuildBuyerPriceCaps()
{
    BuyingBotMaxPricePerItem.clear();
    BuyingBotUndercutMaxPricePerItem.clear();
    SellerSafetyBlockedItemIDs.clear();
    SellerPriceFloorPerItem.clear();

    // Vendor items sold for gold. Unlimited ones and ones restocking at least VendorLoopRestockPerHour (summed over
    // every vendor) feed loops; the slower ones only cap what the buyer pays for the item itself, since a loop
    // through a few items every 12 hours is worth next to nothing
    std::unordered_set<uint32> vendorLoopItems;
    std::unordered_set<uint32> vendorSlowItems;
    std::unordered_set<uint32> vendorUnlimitedItems;
    if (QueryResult result = WorldDatabase.Query("SELECT item, MAX(maxcount = 0), SUM(IF(maxcount > 0 AND incrtime > 0, maxcount * 3600.0 / incrtime, 0)) FROM npc_vendor WHERE item > 0 AND ExtendedCost = 0 GROUP BY item"))
    {
        do
        {
            Field* fields = result->Fetch();
            uint32 itemId = fields[0].Get<uint32>();
            if (fields[1].Get<int64>())
                vendorUnlimitedItems.insert(itemId);
            if (fields[1].Get<int64>() || fields[2].Get<double>() >= VendorLoopRestockPerHour)
                vendorLoopItems.insert(itemId);
            else
                vendorSlowItems.insert(itemId);
        } while (result->NextRow());
    }

    // What the seller bot can list, at the lowest price it could ever ask. Anything it could list below its vendor
    // value stays off the auction house
    std::set<std::pair<uint32, uint32>> listedClassQualities;
    for (ListProportionNode const& node : ItemListProportionNodesLookup)
        listedClassQualities.insert({ node.ItemClassID, node.ItemQualityID });
    std::unordered_map<uint32, double> sellerMinimumPrices;
    uint32 belowVendorValue = 0;
    for (auto const& [itemClass, qualities] : ItemCandidatesByItemClassAndQuality)
        for (auto const& [quality, itemIds] : qualities)
            if (listedClassQualities.count({ itemClass, quality }))
                for (uint32 itemId : itemIds)
                    if (ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId))
                    {
                        uint64 price = GetSellerMinimumPricePerItem(proto);
                        if (price < proto->SellPrice)
                        {
                            SellerSafetyBlockedItemIDs.insert(itemId);
                            ++belowVendorValue;
                        }
                        else
                            sellerMinimumPrices[itemId] = double(price);
                    }

    // Starting ceilings from quality and vendor value
    std::unordered_map<uint32, double> baseCaps;
    for (auto const& [itemId, proto] : *sObjectMgr->GetItemTemplateStore())
    {
        if (proto.Bonding == BIND_WHEN_PICKED_UP || proto.Bonding == BIND_QUEST_ITEM || proto.HasFlag(ITEM_FLAG_CONJURED))
            continue;
        if (vendorUnlimitedItems.count(itemId))
            continue;

        auto fixedPrice = BuyingBotZeroValueItemPrices.find(itemId);
        if (fixedPrice != BuyingBotZeroValueItemPrices.end())
            baseCaps[itemId] = double(fixedPrice->second);
        else if (proto.Quality >= BuyingBotMinQuality && proto.Quality <= BuyingBotMaxQuality && proto.SellPrice > 0)
            baseCaps[itemId] = proto.SellPrice * double(proto.Quality == ITEM_QUALITY_RARE ? BuyingBotMaxVendorValueMultiplierRare : BuyingBotMaxVendorValueMultiplier);
    }

    // Loot-producing processes
    LootTable references = LoadLootTable("reference_loot_template");
    LootTable disenchantLoot = LoadLootTable("disenchant_loot_template");
    LootTable prospectingLoot = LoadLootTable("prospecting_loot_template");
    LootTable millingLoot = LoadLootTable("milling_loot_template");
    LootTable itemLoot = LoadLootTable("item_loot_template");
    LootTable spellLoot = LoadLootTable("spell_loot_template");

    std::unordered_set<uint32> spellsWithLoot;
    for (auto const& [spellId, rows] : spellLoot)
        spellsWithLoot.insert(spellId);
    std::vector<CraftRecipe> recipes = LoadProfessionRecipes(spellsWithLoot);

    std::vector<LootProcess> processes;
    auto addProcess = [&](uint32 sourceItem, uint32 sourceCount, CraftRecipe const* recipe, LootTable const& table, uint32 entry, double money, char const* name)
    {
        LootProcess process{ sourceItem, sourceCount, recipe, {}, money, name };
        AddExpectedLoot(table, entry, 1.0, references, process.Yield);
        if (!process.Yield.empty() || money > 0)
            processes.push_back(std::move(process));
    };
    for (auto const& [itemId, proto] : *sObjectMgr->GetItemTemplateStore())
    {
        if (proto.DisenchantID)
            addProcess(itemId, 1, nullptr, disenchantLoot, proto.DisenchantID, 0.0, "disenchanting");
        if (prospectingLoot.count(itemId))
            addProcess(itemId, 5, nullptr, prospectingLoot, itemId, 0.0, "prospecting");
        if (millingLoot.count(itemId))
            addProcess(itemId, 5, nullptr, millingLoot, itemId, 0.0, "milling");
        if (itemLoot.count(itemId) || proto.MaxMoneyLoot > 0)
            addProcess(itemId, 1, nullptr, itemLoot, itemId, (proto.MinMoneyLoot + proto.MaxMoneyLoot) / 2.0, "opening");
    }
    for (CraftRecipe const& recipe : recipes)
        if (spellsWithLoot.count(recipe.SpellId))
            addProcess(0, 1, &recipe, spellLoot, recipe.SpellId, 0.0, "recipe");

    std::unordered_map<uint32, double> caps;
    std::unordered_map<uint32, double> undercutCaps;
    // Ceilings lowered because a loop from vendor or recipe input produces the item; they bound both ceilings
    std::unordered_map<uint32, double> outputLimits;
    std::unordered_map<uint32, double> sellerFloors;
    uint32 rounds = 0;
    uint32 outputAdjustments = 0;
    bool settled = false;
    while (!settled && rounds < MaxPolicyRounds)
    {
        ++rounds;

        // Cheapest known way to obtain each item, and the cheapest one that is not the seller bot's own listing
        std::unordered_map<uint32, KnownCost> costs;
        std::unordered_map<uint32, double> otherCosts;
        auto offerCost = [&costs, &otherCosts](uint32 itemId, double cost, CostSource source, CraftRecipe const* recipe)
        {
            if (source != CostSource::Seller)
            {
                auto otherItr = otherCosts.find(itemId);
                if (otherItr == otherCosts.end() || cost < otherItr->second)
                    otherCosts[itemId] = cost;
            }
            auto itr = costs.find(itemId);
            if (itr == costs.end() || cost < itr->second.Cost)
            {
                costs[itemId] = { cost, source, recipe };
                return true;
            }
            return false;
        };
        for (uint32 itemId : vendorLoopItems)
            if (ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId))
                offerCost(itemId, proto->BuyPrice * VendorBestDiscount, CostSource::Vendor, nullptr);
        for (auto const& [itemId, price] : sellerMinimumPrices)
        {
            if (SellerSafetyBlockedItemIDs.count(itemId))
                continue;
            auto floorItr = sellerFloors.find(itemId);
            offerCost(itemId, floorItr != sellerFloors.end() ? std::max(price, floorItr->second) : price, CostSource::Seller, nullptr);
        }

        auto recipeCost = [&costs](CraftRecipe const& recipe, double& outCost)
        {
            outCost = 0.0;
            for (auto const& [reagent, count] : recipe.Reagents)
            {
                auto itr = costs.find(reagent);
                if (itr == costs.end())
                    return false;
                outCost += itr->second.Cost * count;
            }
            return true;
        };
        for (uint32 pass = 0; pass < MaxRecipeDepth; ++pass)
        {
            bool improved = false;
            for (CraftRecipe const& recipe : recipes)
            {
                double cost = 0.0;
                if (recipe.CreatedItem && recipeCost(recipe, cost))
                    improved |= offerCost(recipe.CreatedItem, cost / recipe.CreatedCount, CostSource::Craft, &recipe);
            }
            if (!improved)
                break;
        }

        // How many of each seller bot item go into the cheapest way to obtain an item
        std::function<void(uint32, double, std::unordered_map<uint32, double>&, uint32)> collectSellerItems =
            [&](uint32 itemId, double quantity, std::unordered_map<uint32, double>& out, uint32 depth)
        {
            auto itr = costs.find(itemId);
            if (itr == costs.end() || depth > MaxRecipeDepth)
                return;
            if (itr->second.Source == CostSource::Seller)
                out[itemId] += quantity;
            else if (itr->second.Source == CostSource::Craft)
                for (auto const& [reagent, count] : itr->second.Recipe->Reagents)
                    collectSellerItems(reagent, quantity * count / itr->second.Recipe->CreatedCount, out, depth + 1);
        };

        caps.clear();
        undercutCaps.clear();
        for (auto const& [itemId, baseCap] : baseCaps)
        {
            double obtainCap = std::numeric_limits<double>::max();
            auto itr = costs.find(itemId);
            if (itr != costs.end())
                obtainCap = itr->second.Cost * SafetyMargin;
            if (vendorSlowItems.count(itemId))
                if (ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId))
                    obtainCap = std::min(obtainCap, proto->BuyPrice * VendorBestDiscount * SafetyMargin);
            auto limitItr = outputLimits.find(itemId);
            if (limitItr != outputLimits.end())
                obtainCap = std::min(obtainCap, limitItr->second);

            double cap = std::min(baseCap, obtainCap);
            if (cap >= 1.0)
                caps[itemId] = cap;

            // Undercutting auctions: the runtime also keeps them within SafetyMargin of the seller's cheapest live listing
            auto sellerItr = sellerMinimumPrices.find(itemId);
            if (sellerItr == sellerMinimumPrices.end() || SellerSafetyBlockedItemIDs.count(itemId))
                continue;
            auto floorItr = sellerFloors.find(itemId);
            double undercutCap = floorItr != sellerFloors.end() ? std::max(sellerItr->second, floorItr->second) : sellerItr->second;
            auto otherItr = otherCosts.find(itemId);
            if (otherItr != otherCosts.end())
                undercutCap = std::min(undercutCap, otherItr->second * SafetyMargin);
            if (vendorSlowItems.count(itemId))
                if (ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId))
                    undercutCap = std::min(undercutCap, proto->BuyPrice * VendorBestDiscount * SafetyMargin);
            if (limitItr != outputLimits.end())
                undercutCap = std::min(undercutCap, limitItr->second);
            if (undercutCap > cap && undercutCap >= 1.0)
                undercutCaps[itemId] = undercutCap;
        }
        // What the buyer may pay for an item at most, undercutting auctions included
        auto highestCap = [&caps, &undercutCaps](uint32 itemId)
        {
            auto undercutItr = undercutCaps.find(itemId);
            if (undercutItr != undercutCaps.end())
                return undercutItr->second;
            auto capItr = caps.find(itemId);
            return capItr != caps.end() ? capItr->second : 0.0;
        };

        bool changed = false;
        // A loop through the seller bot's items: raise their minimum price just enough that the input costs the target,
        // or keep them off the auction house when that cannot work
        auto raiseSellerFloors = [&](std::unordered_map<uint32, double> const& sellerItems, double inputCost, double target, char const* reason, uint32 itemOrSpell)
        {
            double sellerShare = 0.0;
            for (auto const& [itemId, quantity] : sellerItems)
                sellerShare += quantity * costs[itemId].Cost;
            double otherShare = inputCost - sellerShare;
            double factor = sellerShare > 0.0 ? std::max((target * 1.001 - otherShare) / sellerShare, 1.001) : 0.0;
            bool raise = sellerShare > 0.0 && factor <= MaxSellerFloorRaise;
            for (auto const& [itemId, quantity] : sellerItems)
            {
                if (raise)
                    sellerFloors[itemId] = std::max(sellerFloors[itemId], costs[itemId].Cost * factor);
                else
                    SellerSafetyBlockedItemIDs.insert(itemId);
            }
            changed = true;
            if (debug_Out)
                LOG_INFO("module", "AHBotPolicy: {} {} pays {} for a {} input, {} seller item(s) {}", reason, itemOrSpell, uint64(target), uint64(inputCost),
                    sellerItems.size(), raise ? fmt::format("raised {:.2f}x", factor) : std::string("kept off the auction house"));
        };

        // Crafting from the seller's items and vendoring the result
        for (auto const& [itemId, cost] : costs)
        {
            if (cost.Source != CostSource::Craft)
                continue;
            ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId);
            if (!proto || proto->SellPrice <= cost.Cost * 1.0001)
                continue;
            std::unordered_map<uint32, double> sellerItems;
            collectSellerItems(itemId, 1.0, sellerItems, 0);
            if (!sellerItems.empty())
                raiseSellerFloors(sellerItems, cost.Cost, proto->SellPrice, "vendoring crafted item", itemId);
        }

        for (LootProcess const& process : processes)
        {
            double inputCost = 0.0;
            std::unordered_map<uint32, double> sellerItems;
            if (process.Recipe)
            {
                if (!recipeCost(*process.Recipe, inputCost))
                    continue;
                for (auto const& [reagent, count] : process.Recipe->Reagents)
                    collectSellerItems(reagent, count, sellerItems, 0);
            }
            else
            {
                auto itr = costs.find(process.SourceItem);
                if (itr == costs.end())
                    continue;
                inputCost = itr->second.Cost * process.SourceCount;
                collectSellerItems(process.SourceItem, process.SourceCount, sellerItems, 0);
            }

            double buyerValue = 0.0;
            double vendorValue = process.Money;
            for (auto const& [itemId, count] : process.Yield)
            {
                buyerValue += highestCap(itemId) * count;
                if (ItemTemplate const* proto = sObjectMgr->GetItemTemplate(itemId))
                    vendorValue += proto->SellPrice * count;
            }

            bool buyerLoop = buyerValue > inputCost * SafetyMargin * 1.0001;
            bool vendorLoop = vendorValue > inputCost * 1.0001;
            uint32 itemOrSpell = process.Recipe ? process.Recipe->SpellId : process.SourceItem;
            if (!sellerItems.empty())
            {
                if (buyerLoop || vendorLoop)
                    raiseSellerFloors(sellerItems, inputCost, std::max(buyerValue / SafetyMargin, vendorValue), process.Name, itemOrSpell);
                continue;
            }
            if (!buyerLoop)
                continue;

            // Vendor or recipe input: lower what the buyer pays for the output
            double factor = inputCost * SafetyMargin / buyerValue;
            for (auto const& [itemId, count] : process.Yield)
            {
                double cap = highestCap(itemId);
                if (cap > 0.0)
                {
                    auto limitItr = outputLimits.find(itemId);
                    outputLimits[itemId] = limitItr != outputLimits.end() ? std::min(limitItr->second, cap * factor) : cap * factor;
                }
            }
            changed = true;
            ++outputAdjustments;
            if (debug_Out)
                LOG_INFO("module", "AHBotPolicy: {} {} pays {} for a {} input, output ceilings scaled by {:.3f}",
                    process.Name, itemOrSpell, uint64(buyerValue), uint64(inputCost), factor);
        }

        settled = !changed;
    }

    if (!settled)
    {
        LOG_ERROR("module", "AuctionHouseBot: price policy did not settle in {} rounds, the buyer is disabled", MaxPolicyRounds);
        return;
    }

    for (auto const& [itemId, cap] : caps)
        BuyingBotMaxPricePerItem[itemId] = uint64(cap);
    for (auto const& [itemId, cap] : undercutCaps)
        if (!SellerSafetyBlockedItemIDs.count(itemId))
            BuyingBotUndercutMaxPricePerItem[itemId] = uint64(cap);
    for (auto const& [itemId, floor] : sellerFloors)
        if (!SellerSafetyBlockedItemIDs.count(itemId) && floor > sellerMinimumPrices[itemId])
            SellerPriceFloorPerItem[itemId] = uint64(std::ceil(floor));

    if (debug_Out)
    {
        for (auto const& [itemId, listedPrice] : BuyingBotZeroValueItemPrices)
        {
            auto capItr = BuyingBotMaxPricePerItem.find(itemId);
            LOG_INFO("module", "AHBotPolicy: zero-value item {} listed at {}, buyer pays at most {}", itemId, listedPrice,
                capItr != BuyingBotMaxPricePerItem.end() ? capItr->second : 0);
        }
        for (auto const& [itemId, floor] : SellerPriceFloorPerItem)
            LOG_INFO("module", "AHBotPolicy: seller item {} minimum price {} (was {})", itemId, floor, uint64(sellerMinimumPrices[itemId]));
        for (uint32 itemId : SellerSafetyBlockedItemIDs)
            LOG_INFO("module", "AHBotPolicy: seller item {} kept off the auction house", itemId);
    }

    LOG_INFO("module", "AuctionHouseBot: price policy settled in {} rounds: buyer ceilings for {} items ({} higher for undercutting auctions), {} output ceilings lowered, {} seller items with a raised minimum price, {} kept off the auction house ({} below vendor value), {} recipes and {} loot processes checked",
        rounds, BuyingBotMaxPricePerItem.size(), BuyingBotUndercutMaxPricePerItem.size(), outputAdjustments, SellerPriceFloorPerItem.size(), SellerSafetyBlockedItemIDs.size(), belowVendorValue, recipes.size(), processes.size());
}
